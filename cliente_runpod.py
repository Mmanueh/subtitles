"""
Cliente para llamar a un endpoint de RunPod Serverless que ejecuta el
script de traducción/mejora de subtítulos.

Uso:
    # Síncrono (espera hasta que termine, ideal para archivos chicos):
    python cliente_runpod.py "video.srt"

    # Asíncrono (devuelve job_id inmediatamente, recomendado para SRTs largos):
    python cliente_runpod.py "video.srt" --async

    # Especificar idiomas / modelo:
    python cliente_runpod.py "video.srt" -s Japanese -t Spanish -m qwen2.5:7b

Variables de entorno:
    RUNPOD_API_KEY        Tu API key de RunPod (obligatoria)
    RUNPOD_ENDPOINT_ID    El ID de tu endpoint serverless (obligatoria)
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _post(url: str, api_key: str, payload: dict, timeout: float = 600.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _get(url: str, api_key: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def llamar_sincrono(
    endpoint_id: str,
    api_key: str,
    payload: dict,
    timeout: float = 600.0,
) -> dict:
    """Usa /runsync — bloquea hasta que termine o llegue al timeout del worker."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    return _post(url, api_key, {"input": payload}, timeout=timeout)


def llamar_asincrono(endpoint_id: str, api_key: str, payload: dict) -> str:
    """Encola el job y devuelve job_id."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    resp = _post(url, api_key, {"input": payload})
    job_id = resp.get("id")
    if not job_id:
        raise RuntimeError(f"Respuesta sin id: {resp}")
    return job_id


def consultar_estado(endpoint_id: str, api_key: str, job_id: str) -> dict:
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    return _get(url, api_key)


def esperar_a_que_termine(
    endpoint_id: str,
    api_key: str,
    job_id: str,
    intervalo: float = 3.0,
    timeout: float = 1800.0,
) -> dict:
    """Polling cada N segundos hasta que el job esté COMPLETED o FAILED."""
    t0 = time.time()
    while True:
        estado = consultar_estado(endpoint_id, api_key, job_id)
        status = estado.get("status", "UNKNOWN")
        print(f"   [{int(time.time() - t0):4d}s] estado={status}")
        if status in ("COMPLETED", "FAILED", "CANCELLED"):
            return estado
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Job {job_id} no terminó en {timeout}s")
        time.sleep(intervalo)


def construir_payload(args: argparse.Namespace, contenido_srt: str) -> dict:
    payload = {
        "srt": contenido_srt,
        "idioma_origen": args.idioma_origen,
        "idioma_destino": args.idioma_destino,
        "modelo": args.modelo,
        "chunk": args.chunk,
        "temperatura": args.temperatura,
        "max_chars_por_linea": args.max_chars_por_linea,
        "max_lineas": args.max_lineas,
    }
    if args.prompt_file:
        payload["prompt"] = Path(args.prompt_file).read_text(encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cliente para el endpoint RunPod.")
    p.add_argument("srt", help="Ruta al .srt de entrada.")
    p.add_argument("--output", "-o", help="Ruta del .srt de salida.")
    p.add_argument("--async", dest="asincrono", action="store_true",
                   help="Usar /run (asíncrono con polling) en vez de /runsync.")
    p.add_argument("--endpoint-id",
                   default=os.environ.get("RUNPOD_ENDPOINT_ID"),
                   help="ID del endpoint serverless. "
                        "Default: $RUNPOD_ENDPOINT_ID.")
    p.add_argument("--api-key",
                   default=os.environ.get("RUNPOD_API_KEY"),
                   help="API key de RunPod. Default: $RUNPOD_API_KEY.")
    p.add_argument("--idioma-origen", "-s", default="Japanese")
    p.add_argument("--idioma-destino", "-t", default="English")
    p.add_argument("--modelo", "-m", default="qwen3.5:32b",
                   help="Modelo de Ollama. Ej: qwen3.5:32b, vanilj/mistral-nemo-12b-celeste-v1.9")
    p.add_argument("--chunk", type=int, default=30)
    p.add_argument("--temperatura", type=float, default=0.6)
    p.add_argument("--max-chars-por-linea", type=int, default=50)
    p.add_argument("--max-lineas", type=int, default=2)
    p.add_argument("--prompt-file", help="Archivo con prompt de sistema custom.")
    p.add_argument("--timeout", type=float, default=600.0,
                   help="Timeout en segundos para /runsync. Default: 600.")
    p.add_argument("--poll-interval", type=float, default=3.0,
                   help="Intervalo de polling en modo async. Default: 3s.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.api_key:
        print("❌ Falta RUNPOD_API_KEY (variable de entorno o --api-key).",
              file=sys.stderr)
        return 1
    if not args.endpoint_id:
        print("❌ Falta RUNPOD_ENDPOINT_ID (variable de entorno o --endpoint-id).",
              file=sys.stderr)
        return 1

    ruta_in = Path(args.srt).expanduser().resolve()
    if not ruta_in.exists():
        print(f"❌ No existe: {ruta_in}", file=sys.stderr)
        return 1

    contenido = ruta_in.read_text(encoding="utf-8-sig")
    payload = construir_payload(args, contenido)
    ruta_out = Path(args.output).expanduser().resolve() if args.output else \
        ruta_in.with_name(ruta_in.stem + ".translated.srt")

    print(f"📤 Enviando {ruta_in.name} ({len(contenido)} caracteres) al endpoint {args.endpoint_id}")
    print(f"   modo: {'async + polling' if args.asincrono else 'sync'}")
    t_inicio = time.time()

    try:
        if args.asincrono:
            job_id = llamar_asincrono(args.endpoint_id, args.api_key, payload)
            print(f"   job_id: {job_id}")
            print("⏳ Esperando resultado (polling)...")
            estado = esperar_a_que_termine(
                args.endpoint_id, args.api_key, job_id,
                intervalo=args.poll_interval,
            )
            if estado.get("status") != "COMPLETED":
                print(f"❌ Job no completado: {json.dumps(estado, indent=2)}",
                      file=sys.stderr)
                return 2
            output = estado.get("output", {})
        else:
            print("⏳ Esperando (sync)...")
            resp = llamar_sincrono(
                args.endpoint_id, args.api_key, payload,
                timeout=args.timeout,
            )
            # /runsync devuelve {"id": ..., "status": "COMPLETED", "output": {...}}
            # o bien con FAILED + error
            if resp.get("status") not in ("COMPLETED", None):
                print(f"❌ Job falló: {json.dumps(resp, indent=2)}", file=sys.stderr)
                return 2
            output = resp.get("output", {})
    except urllib.error.HTTPError as e:
        print(f"❌ HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}",
              file=sys.stderr)
        return 2
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    if isinstance(output, dict) and "error" in output:
        print(f"❌ Error del worker: {output['error']}", file=sys.stderr)
        return 2

    srt_resultado = output.get("srt") if isinstance(output, dict) else None
    if not srt_resultado:
        print(f"❌ Sin SRT en la respuesta: {json.dumps(output, indent=2)[:500]}",
              file=sys.stderr)
        return 2

    ruta_out.write_text(srt_resultado, encoding="utf-8")
    dur = time.time() - t_inicio

    stats = output.get("stats", {}) if isinstance(output, dict) else {}
    print(f"\n✅ Listo en {dur:.1f}s")
    if stats:
        print(f"   Subtítulos: {stats.get('subs_entrada', '?')} → "
              f"{stats.get('subs_salida', '?')}")
        print(f"   Modelo: {stats.get('modelo_usado', '?')}")
        print(f"   Duración del worker: {stats.get('duracion_s', '?')}s")
    print(f"📄 Guardado: {ruta_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

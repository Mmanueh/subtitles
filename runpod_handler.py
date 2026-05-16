"""
Handler de RunPod Serverless para traducir/mejorar archivos SRT.

Este script vive dentro del contenedor que sube a RunPod. RunPod invoca
`handler(job)` cuando llega una petición.

Entrada esperada (job["input"]):
  {
    "srt": "<contenido del SRT como string>",        # obligatorio (uno u otro)
    "srt_url": "https://...",                        #   "
    "idioma_origen": "Japanese",                     # opcional, default Japanese
    "idioma_destino": "English",                     # opcional, default English
    "modelo": "qwen2.5:14b",                         # opcional
    "chunk": 15,                                     # opcional
    "temperatura": 0.3,                              # opcional
    "max_chars_por_linea": 42,                       # opcional
    "max_lineas": 2,                                 # opcional
    "prompt": "<prompt de sistema custom>"           # opcional
  }

Salida:
  {
    "srt": "<contenido del SRT traducido>",
    "stats": {
      "subs_entrada": int,
      "subs_salida": int,
      "alucinaciones_descartadas": int,
      "chunks_idioma_malo": int,
      "duracion_s": float
    }
  }
"""

import os
import subprocess
import sys
import time
import tempfile
import urllib.request
from pathlib import Path

import runpod

# Importamos las funciones del script principal.
# Asegúrate de que mejorar_subtitulos.py esté en el mismo directorio.
sys.path.insert(0, str(Path(__file__).parent))
import mejorar_subtitulos as ms


# ---------- Arranque del servidor Ollama dentro del contenedor ----------

_ollama_proc = None


def asegurar_ollama_arrancado(modelo: str) -> None:
    """
    Arranca el daemon de Ollama si no está corriendo y se asegura de que
    el modelo esté disponible. Se llama solo en el primer request del worker
    (cold start). En requests subsiguientes ya estará todo cargado.
    """
    global _ollama_proc

    # ¿Ya está corriendo?
    try:
        import ollama
        ollama.Client(timeout=2).list()
        # Sí está, verificamos el modelo
        _verificar_modelo(modelo)
        return
    except Exception:
        pass  # No está, hay que arrancarlo

    print("🚀 Arrancando daemon de Ollama...")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = "127.0.0.1:11434"
    # Almacenamiento del modelo: usa el volumen montado por RunPod si existe
    if Path("/runpod-volume").exists():
        env["OLLAMA_MODELS"] = "/runpod-volume/ollama"
        Path("/runpod-volume/ollama").mkdir(exist_ok=True)

    _ollama_proc = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Esperar a que esté listo (max 60s)
    import ollama
    for _ in range(60):
        try:
            ollama.Client(timeout=2).list()
            print("   Ollama listo.")
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError("Ollama no arrancó en 60s")

    _verificar_modelo(modelo)


def _verificar_modelo(modelo: str) -> None:
    """Si el modelo no está descargado, lo descarga."""
    import ollama
    client = ollama.Client(timeout=600)
    try:
        modelos = [m.get("model", m.get("name", "")) for m in client.list().get("models", [])]
        if any(modelo in m or m.startswith(modelo.split(":")[0]) for m in modelos):
            return
    except Exception:
        pass

    print(f"📥 Descargando modelo {modelo}... (esto puede tardar varios minutos)")
    t0 = time.time()
    client.pull(modelo)
    print(f"   Modelo listo en {time.time() - t0:.0f}s")


# ---------- Helpers de I/O ----------

def _obtener_srt(job_input: dict) -> str:
    """Devuelve el contenido del SRT, ya sea inline o descargado de URL."""
    if "srt" in job_input and job_input["srt"]:
        return job_input["srt"]
    if "srt_url" in job_input and job_input["srt_url"]:
        with urllib.request.urlopen(job_input["srt_url"], timeout=60) as r:
            data = r.read()
        # SRT puede venir con BOM
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace")
    raise ValueError("Falta 'srt' (contenido) o 'srt_url' en input.")


# ---------- Handler principal ----------

def handler(job: dict) -> dict:
    """
    RunPod invoca esta función por cada petición.
    `job` tiene la forma {"id": "...", "input": {...}}.
    """
    t_inicio = time.time()
    inp = job.get("input", {}) or {}

    # --- Parámetros con defaults sensatos
    # --- Parámetros optimizados para Qwen3.5-37B ---
    idioma_origen = inp.get("idioma_origen", "Japanese")
    idioma_destino = inp.get("idioma_destino", "English")
    modelo = inp.get("modelo", "qwen3.5:32b")
    chunk = int(inp.get("chunk", 30))                   # Más contexto = mejor calidad
    temperatura = float(inp.get("temperatura", 0.6))    # Qwen responde bien con 0.55-0.7
    num_ctx = int(inp.get("num_ctx", 32768))            # Qwen maneja bien contexto largo
    max_chars = int(inp.get("max_chars_por_linea", 50))
    timeout_chat = float(inp.get("timeout", 900.0))
    reintentos = int(inp.get("reintentos", 2))
    max_lineas = int(inp.get("max_lineas", 2))
    prompt_sistema = inp.get("prompt") or ms.PROMPT_POR_DEFECTO

    # --- Asegurar que Ollama y el modelo estén listos
    asegurar_ollama_arrancado(modelo)

    # --- Leer el SRT a un archivo temporal (mejorar_srt espera Paths)
    contenido_srt = _obtener_srt(inp)

    with tempfile.TemporaryDirectory() as tmpdir:
        ruta_in = Path(tmpdir) / "input.srt"
        ruta_out = Path(tmpdir) / "output.srt"
        ruta_in.write_text(contenido_srt, encoding="utf-8")

        try:
            ms.mejorar_srt(
                ruta_entrada=ruta_in,
                ruta_salida=ruta_out,
                idioma_origen=idioma_origen,
                idioma_destino=idioma_destino,
                modelo=modelo,
                prompt_sistema=prompt_sistema,
                chunk_size=chunk,
                temperatura=temperatura,
                num_ctx=num_ctx,
                timeout=timeout_chat,
                reintentos=reintentos,
                host="http://127.0.0.1:11434",
                max_chars_por_linea=max_chars,
                max_lineas=max_lineas,
            )
        except Exception as e:
            return {
                "error": f"{type(e).__name__}: {e}",
                "duracion_s": round(time.time() - t_inicio, 1),
            }

        srt_resultado = ruta_out.read_text(encoding="utf-8")

    # Contar subtítulos para devolver stats útiles
    subs_entrada = sum(1 for line in contenido_srt.split("\n\n") if line.strip())
    subs_salida = sum(1 for line in srt_resultado.split("\n\n") if line.strip())

    return {
        "srt": srt_resultado,
        "stats": {
            "subs_entrada": subs_entrada,
            "subs_salida": subs_salida,
            "modelo_usado": modelo,
            "duracion_s": round(time.time() - t_inicio, 1),
        },
    }


if __name__ == "__main__":
    # RunPod arranca el worker llamando a esta función.
    runpod.serverless.start({"handler": handler})

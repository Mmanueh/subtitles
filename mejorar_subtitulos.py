"""
Traduce y mejora un archivo de subtítulos (.srt) usando un modelo de Ollama
(qwen2.5 por defecto).

Uso:
    python mejorar_subtitulos.py "ruta/al/subtitulos.srt"
    python mejorar_subtitulos.py "video.ja.srt" --idioma-origen Japanese --idioma-destino English
    python mejorar_subtitulos.py "video.srt" --modelo qwen2.5:7b --chunk 20
    python mejorar_subtitulos.py "video.srt" --prompt-file mi_prompt.txt

El archivo de salida se genera en la misma carpeta. Si la entrada es
`video.ja.srt` y el destino es English, la salida será `video.en.srt`.
"""

import argparse
import json
import re
import sys
import time
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import ollama


# ============== FILTRO DE ALUCINACIONES DE WHISPER ==============
# Detecta líneas basura típicas de Whisper antes de traducirlas: ahorra tokens
# y evita que el LLM intente "traducir" う ぅ う ぅ う ぅ...

_PUNTUACION_RE = re.compile(
    r'[\s、。,.!?！？…ー〜・「」『』（）()【】\u3000]+'
)


def _solo_letras(s: str) -> str:
    return _PUNTUACION_RE.sub('', s)


def _es_alucinacion(texto: str) -> tuple[bool, str]:
    t = texto.strip()
    if not t:
        return False, ""
    chars = _solo_letras(t)
    if not chars:
        return False, ""
    if len(chars) >= 8:
        contador = Counter(chars)
        char_top, freq = contador.most_common(1)[0]
        if freq / len(chars) >= 0.80:
            return True, f"caracter '{char_top}' ocupa {int(100*freq/len(chars))}%"
    partes = [p.strip() for p in re.split(r'[、,。.]+', t) if p.strip()]
    if len(partes) >= 4:
        contador_p = Counter(partes)
        parte_top, freq_p = contador_p.most_common(1)[0]
        if freq_p / len(partes) >= 0.75 and len(parte_top) <= 8:
            return True, f"frase '{parte_top}' repetida {freq_p}/{len(partes)}"
    if len(t) >= 30:
        bytes_t = t.encode('utf-8')
        comprimido = zlib.compress(bytes_t)
        ratio_comp = len(bytes_t) / max(1, len(comprimido))
        if ratio_comp > 6:
            return True, f"compresión {ratio_comp:.1f}x"
    return False, ""
# =================================================================


# ================== PROMPT POR DEFECTO ==================
# Edita libremente esta plantilla. Los marcadores {idioma_origen} y
# {idioma_destino} se reemplazan automáticamente.
# También puedes guardar tu prompt en un archivo y pasarlo con --prompt-file.
PROMPT_POR_DEFECTO = """You are a professional JAV erotic scriptwriter and dialogue enhancer.

You will receive a list of subtitle lines in {idioma_origen}.

For each line, create a greatly enhanced English version that:
- Dramatically increases sexual explicitness and heat.
- Makes female characters extremely horny, eager, seductive, and proactive.
- Converts any hesitation or shyness into a mix between hesitation or shynes with strong lust and encouragement.
- NEVER leave Japanese words (manko, chinpo, etc.). Always translate to strong English.
- In taboo scenarios (mother/son, teacher/student, etc.), make the woman actively initiate and beg for more.
- Sound like raw, high-quality English JAV / porn dialogue.
- NEVER use euphemisms.
- Maintain PERFECT natural English grammar, conjugation and sentence structure.
- Every line must be a complete, natural sounding English sentence or exclamation.
- Do not use fragments or missing subjects.
- Make the dialogue flow naturally as if a native English speaker is saying it in porn.

CRITICAL RULES:
- Be explicit and vulgar. Do not tone down or avoid sexual language.
- Make the woman sound enthusiastic and desperate for sex.
- Keep lines concise: aim for under {max_chars_total} characters per line.
- Maintain correct speaker and context.

Respond with a single JSON object EXACTLY in this shape:
{{
  "lines": [
    {{"index": <integer>, "text": "<enhanced erotic English text>"}},
    ...
  ]
}}

Rules:
- Return exactly ONE entry per input line, with the SAME index value.
- "text" must be a single line (no embedded newlines).
- EVERYTHING must be in natural, fluent English. No Japanese words allowed.
- Do not add any explanations or comments.
"""
# ========================================================


# Mapeo idioma → código corto para el nombre de archivo de salida
MAPA_CODIGOS = {
    "english": "en", "spanish": "es", "japanese": "ja", "french": "fr",
    "german": "de", "portuguese": "pt", "italian": "it", "korean": "ko",
    "chinese": "zh", "russian": "ru", "arabic": "ar", "dutch": "nl",
}


@dataclass
class Subtitulo:
    indice: int
    inicio: str   # "HH:MM:SS,mmm"
    fin: str
    texto: str


# ---------------------- SRT I/O ----------------------

_TIMESTAMP_RE = re.compile(
    r"(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
)


def parse_srt(ruta: Path) -> list[Subtitulo]:
    """Parser tolerante de archivos .srt."""
    contenido = ruta.read_text(encoding="utf-8-sig")
    contenido = contenido.replace("\r\n", "\n").replace("\r", "\n")
    bloques = re.split(r"\n\s*\n", contenido.strip())
    subs: list[Subtitulo] = []

    for bloque in bloques:
        lineas = [l for l in bloque.split("\n") if l.strip() != ""]
        if len(lineas) < 2:
            continue

        # La primera línea suele ser el índice; si no, asignamos uno secuencial.
        try:
            idx = int(lineas[0].strip())
            ts_line = lineas[1] if len(lineas) > 1 else ""
            texto_lineas = lineas[2:]
        except ValueError:
            idx = len(subs) + 1
            ts_line = lineas[0]
            texto_lineas = lineas[1:]

        m = _TIMESTAMP_RE.search(ts_line)
        if not m:
            continue
        inicio = m.group(1).replace(".", ",")
        fin = m.group(2).replace(".", ",")
        # Subtítulos multilínea se unen en una sola (lo común tras traducir).
        texto = " ".join(l.strip() for l in texto_lineas).strip()
        subs.append(Subtitulo(idx, inicio, fin, texto))

    return subs


def write_srt(ruta: Path, subs: Iterable[Subtitulo]) -> None:
    """Escribe SRT de forma atómica (tmp + rename)."""
    tmp = ruta.with_suffix(ruta.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for s in subs:
                f.write(f"{s.indice}\n{s.inicio} --> {s.fin}\n{s.texto}\n\n")
        tmp.replace(ruta)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


# ---------------------- Wrap / Split de líneas largas ----------------------

def ts_a_segundos(ts: str) -> float:
    """'HH:MM:SS,mmm' → float segundos."""
    h, m, resto = ts.split(":")
    s, ms = resto.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def segundos_a_ts(seg: float) -> str:
    if seg < 0:
        seg = 0.0
    h = int(seg // 3600)
    m = int((seg % 3600) // 60)
    s = int(seg % 60)
    ms = int(round((seg - int(seg)) * 1000))
    if ms == 1000:
        ms = 0
        s += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def envolver_o_dividir(
    sub: Subtitulo,
    max_chars: int,
    max_lines: int,
) -> list[Subtitulo]:
    """
    Si el texto cabe en `max_lines` líneas de `max_chars` caracteres,
    devuelve un único subtítulo con los saltos de línea adecuados.
    Si no cabe, divide en varios subtítulos repartiendo el tiempo
    proporcionalmente al número de caracteres.
    """
    import textwrap
    texto = " ".join(sub.texto.split())  # normaliza espacios

    if len(texto) <= max_chars:
        return [Subtitulo(sub.indice, sub.inicio, sub.fin, texto)]

    lineas = textwrap.wrap(
        texto,
        width=max_chars,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [texto]

    if len(lineas) <= max_lines:
        # Cabe en un solo subtítulo con varias líneas
        return [Subtitulo(sub.indice, sub.inicio, sub.fin, "\n".join(lineas))]

    # No cabe: dividimos en varios subtítulos de `max_lines` líneas
    grupos = [lineas[i:i + max_lines] for i in range(0, len(lineas), max_lines)]

    inicio_s = ts_a_segundos(sub.inicio)
    fin_s = ts_a_segundos(sub.fin)
    duracion = max(0.001, fin_s - inicio_s)
    total_chars = sum(sum(len(l) for l in g) for g in grupos) or 1

    resultados: list[Subtitulo] = []
    cursor = inicio_s
    for i, grupo in enumerate(grupos):
        chars = sum(len(l) for l in grupo)
        dur_grupo = duracion * (chars / total_chars)
        nuevo_inicio = cursor
        nuevo_fin = fin_s if i == len(grupos) - 1 else cursor + dur_grupo
        cursor = nuevo_fin
        resultados.append(Subtitulo(
            indice=sub.indice,  # se reindexa al final
            inicio=segundos_a_ts(nuevo_inicio),
            fin=segundos_a_ts(nuevo_fin),
            texto="\n".join(grupo),
        ))
    return resultados


def aplicar_post_procesado(
    subs: list[Subtitulo],
    max_chars: int,
    max_lines: int,
) -> tuple[list[Subtitulo], int, int]:
    """
    Aplica wrap/split a una lista de subtítulos y reindexa los resultados.
    Devuelve (lista_resultado, n_envueltos, n_divididos).
    """
    if max_chars <= 0:
        return subs, 0, 0
    salida: list[Subtitulo] = []
    n_envueltos = 0
    n_divididos = 0
    for s in subs:
        partes = envolver_o_dividir(s, max_chars, max_lines)
        if len(partes) > 1:
            n_divididos += 1
        elif "\n" in partes[0].texto and "\n" not in s.texto:
            n_envueltos += 1
        salida.extend(partes)
    # Reindexar
    salida = [
        Subtitulo(i, p.inicio, p.fin, p.texto)
        for i, p in enumerate(salida, 1)
    ]
    return salida, n_envueltos, n_divididos


# ---------------------- Ollama ----------------------

def hacer_chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def procesar_chunk(
    client: ollama.Client,
    chunk: list[Subtitulo],
    modelo: str,
    prompt_sistema: str,
    temperatura: float,
    num_ctx: int,
) -> dict[int, str]:
    """Envía un chunk a Ollama y devuelve {indice: texto_traducido}."""
    payload = {
        "lines": [{"index": s.indice, "text": s.texto} for s in chunk]
    }
    user_msg = (
        "Translate and improve the following subtitle lines. "
        "Respond strictly with the JSON format described in the system prompt.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    response = client.chat(
        model=modelo,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": user_msg},
        ],
        options={
            "temperature": temperatura,
            "num_ctx": num_ctx,
        },
        format="json",  # Fuerza salida JSON válida
    )
    content = response["message"]["content"]
    data = json.loads(content)

    resultado: dict[int, str] = {}
    for item in data.get("lines", []):
        try:
            idx = int(item["index"])
            txt = str(item["text"]).strip()
            # Quitar saltos de línea internos por si el modelo los metió.
            txt = " ".join(txt.split())
            resultado[idx] = txt
        except (KeyError, ValueError, TypeError):
            continue
    return resultado


# ---------------------- Detección de idioma incorrecto ----------------------

class IdiomaIncorrectoError(Exception):
    """El modelo respondió en un idioma distinto al solicitado."""


_LATIN_SCRIPTS = {
    "english", "spanish", "español", "french", "français", "german", "deutsch",
    "italian", "italiano", "portuguese", "português", "dutch", "polish",
    "swedish", "norwegian", "danish", "finnish", "romanian", "czech", "turkish",
}
_CJK_OK_TARGETS = {
    "chinese", "mandarin", "cantonese", "chino", "japanese", "japonés",
    "korean", "coreano",
}


def _texto_es_idioma_incorrecto(texto: str, idioma_destino: str) -> bool:
    """
    Heurística rápida: si el destino usa alfabeto latino pero el texto está
    lleno de caracteres CJK (chino/japonés/coreano), está claramente mal.
    """
    destino = idioma_destino.strip().lower()
    if destino in _CJK_OK_TARGETS:
        # Para destinos CJK no aplicamos esta detección (compleja distinguir
        # chino vs japonés sin librerías extra).
        return False
    if destino not in _LATIN_SCRIPTS:
        # Idioma desconocido para nosotros — no asumimos
        return False

    # Contamos CJK Unified Ideographs + Hiragana + Katakana + Hangul
    cjk_count = 0
    total_letras = 0
    for c in texto:
        if c.isspace() or not c.isprintable():
            continue
        total_letras += 1
        cp = ord(c)
        if (0x4E00 <= cp <= 0x9FFF        # CJK Unified Ideographs (hanzi/kanji)
                or 0x3040 <= cp <= 0x309F  # Hiragana
                or 0x30A0 <= cp <= 0x30FF  # Katakana
                or 0xAC00 <= cp <= 0xD7AF):  # Hangul
            cjk_count += 1

    if total_letras < 3:
        return False  # texto muy corto, no fiable
    # Si más del 30% del texto es CJK siendo destino latino, está mal
    return cjk_count / total_letras > 0.30


def _fraccion_idioma_incorrecto(
    traducciones: dict[int, str], idioma_destino: str
) -> float:
    if not traducciones:
        return 0.0
    n = sum(
        1 for t in traducciones.values()
        if _texto_es_idioma_incorrecto(t, idioma_destino)
    )
    return n / len(traducciones)


# ---------------------- Orquestación ----------------------

def mejorar_srt(
    ruta_entrada: Path,
    ruta_salida: Path,
    idioma_origen: str,
    idioma_destino: str,
    modelo: str,
    prompt_sistema: str,
    chunk_size: int,
    temperatura: float,
    num_ctx: int,
    timeout: float,
    reintentos: int,
    host: str | None,
    max_chars_por_linea: int,
    max_lineas: int,
) -> None:
    print(f"📄 Entrada : {ruta_entrada}")
    print(f"📄 Salida  : {ruta_salida}")
    print(f"🧠 Modelo  : {modelo}")
    print(f"🌐 {idioma_origen} → {idioma_destino} | chunk={chunk_size} "
          f"| temp={temperatura} | num_ctx={num_ctx}")

    subs = parse_srt(ruta_entrada)
    if not subs:
        raise ValueError("El archivo .srt no contiene subtítulos válidos.")
    print(f"📚 {len(subs)} subtítulos cargados.")

    # Pre-filtro: eliminar alucinaciones de Whisper antes de gastar tokens
    subs_limpios: list[Subtitulo] = []
    descartados = 0
    motivos_muestra: list[str] = []
    for s in subs:
        es_h, motivo = _es_alucinacion(s.texto)
        if es_h:
            descartados += 1
            if len(motivos_muestra) < 5:
                motivos_muestra.append(f"#{s.indice} @ {s.inicio[:8]}: {motivo}")
        else:
            subs_limpios.append(s)
    # Re-indexar para que la traducción no tenga huecos
    subs_limpios = [
        Subtitulo(i, x.inicio, x.fin, x.texto)
        for i, x in enumerate(subs_limpios, 1)
    ]
    if descartados:
        print(f"🧹 Pre-filtro: {descartados} líneas descartadas como alucinaciones:")
        for m in motivos_muestra:
            print(f"   - {m}")
        if descartados > len(motivos_muestra):
            print(f"   ...y {descartados - len(motivos_muestra)} más")
    subs = subs_limpios
    print(f"📚 {len(subs)} subtítulos a traducir.\n")

    # Reemplazo seguro de placeholders (no usamos .format para no chocar con {} del JSON)
    max_chars_total = max_chars_por_linea * max_lineas
    prompt_final = (
        prompt_sistema
        .replace("{idioma_origen}", idioma_origen)
        .replace("{idioma_destino}", idioma_destino)
        .replace("{max_chars_total}", str(max_chars_total))
    )

    client = ollama.Client(host=host, timeout=timeout) if host else ollama.Client(timeout=timeout)

    # Warmup: precarga el modelo en VRAM. Usamos el MISMO num_ctx que las
    # peticiones reales para evitar que Ollama tenga que recargar el modelo
    # con otra configuración (eso causa el error "llama runner terminated").
    print("🔥 Precargando modelo en memoria...", end="", flush=True)
    t_warm = time.time()
    try:
        client.chat(
            model=modelo,
            messages=[{"role": "user", "content": "ok"}],
            options={"temperature": 0.0, "num_ctx": num_ctx},
        )
        print(f" listo en {time.time() - t_warm:.1f}s")
    except Exception as e:
        # Si falla el warmup, avisamos pero seguimos: puede que el primer chunk sí funcione.
        print(f" advertencia: {type(e).__name__}: {e}")
        print(f"   (Si persiste, prueba un modelo más pequeño: -m qwen2.5:7b)")
    print()

    mejorados: list[Subtitulo] = []
    fallidos_total = 0
    chunks_idioma_malo = 0
    t_inicio = time.time()

    for n_chunk, chunk in enumerate(hacer_chunks(subs, chunk_size), 1):
        rango = f"{chunk[0].indice}-{chunk[-1].indice}"
        print(f"  ⏳ Chunk {n_chunk} ({rango})...", end="", flush=True)

        traducciones: dict[int, str] = {}
        intento = 0
        while intento <= reintentos:
            try:
                # Sube ligeramente la temperatura en cada reintento para
                # "sacar" al modelo de un patrón malo (como salir en chino).
                temp_intento = temperatura + 0.15 * intento
                traducciones = procesar_chunk(
                    client=client,
                    chunk=chunk,
                    modelo=modelo,
                    prompt_sistema=prompt_final,
                    temperatura=temp_intento,
                    num_ctx=num_ctx,
                )
                # Validar idioma. Si una proporción significativa sale en
                # idioma incorrecto, forzamos reintento.
                frac_mala = _fraccion_idioma_incorrecto(traducciones, idioma_destino)
                if frac_mala > 0.40:
                    raise IdiomaIncorrectoError(
                        f"{int(frac_mala*100)}% de las líneas en idioma incorrecto"
                    )
                break
            except IdiomaIncorrectoError as e:
                intento += 1
                if intento > reintentos:
                    print(f" ⚠️  idioma incorrecto ({e}), uso originales", end="")
                    chunks_idioma_malo += 1
                    traducciones = {}  # forzar fallback
                    break
                print(f" idioma incorrecto, reintento {intento}/{reintentos}...",
                      end="", flush=True)
                time.sleep(0.5)
            except (json.JSONDecodeError, KeyError) as e:
                intento += 1
                if intento > reintentos:
                    print(f" ⚠️  formato inválido ({type(e).__name__}), uso originales", end="")
                    break
                print(f" reintento {intento}/{reintentos}...", end="", flush=True)
                time.sleep(0.8 * intento)
            except Exception as e:
                intento += 1
                if intento > reintentos:
                    print(f" ⚠️  error: {type(e).__name__}: {e}", end="")
                    break
                print(f" reintento {intento}/{reintentos}...", end="", flush=True)
                time.sleep(1.5 * intento)

        # Aplicar traducciones; los que falten conservan el texto original
        ok = 0
        for s in chunk:
            nuevo_texto = traducciones.get(s.indice)
            if nuevo_texto:
                ok += 1
                mejorados.append(Subtitulo(s.indice, s.inicio, s.fin, nuevo_texto))
            else:
                mejorados.append(s)  # fallback al original
        fallidos = len(chunk) - ok
        fallidos_total += fallidos
        print(f" ✓ {ok}/{len(chunk)}" + (f" ({fallidos} fallback)" if fallidos else ""))

    # Post-procesado: envolver / dividir líneas que excedan el límite visual
    mejorados_final, n_envueltos, n_divididos = aplicar_post_procesado(
        mejorados,
        max_chars=max_chars_por_linea,
        max_lines=max_lineas,
    )
    if n_envueltos or n_divididos:
        print(f"✂️  Post-procesado: {n_envueltos} envueltas en 2 líneas, "
              f"{n_divididos} divididas en varios subtítulos.")
        if len(mejorados_final) != len(mejorados):
            print(f"   ({len(mejorados)} → {len(mejorados_final)} entradas tras dividir)")

    write_srt(ruta_salida, mejorados_final)
    dur = time.time() - t_inicio
    print(f"\n✅ Listo en {dur:.1f}s")
    if fallidos_total:
        print(f"⚠️  {fallidos_total} líneas usaron el texto original (fallback).")
    if chunks_idioma_malo:
        print(f"⚠️  {chunks_idioma_malo} chunk(s) salieron en idioma incorrecto "
              "tras todos los reintentos.")
        print("   Si pasa seguido con qwen2.5, prueba un modelo con menos sesgo "
              "hacia el chino:")
        print("   - llama3.1:8b   (más fiable para destino en inglés/español)")
        print("   - mistral:7b    (alternativa ligera)")
    print(f"📄 Archivo generado: {ruta_salida}")


# ---------------------- CLI ----------------------

def cargar_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).expanduser().read_text(encoding="utf-8")
    return PROMPT_POR_DEFECTO


def construir_ruta_salida(
    ruta_entrada: Path,
    idioma_destino: str,
    output_arg: str | None,
) -> Path:
    if output_arg:
        return Path(output_arg).expanduser().resolve()

    codigo = MAPA_CODIGOS.get(idioma_destino.strip().lower(),
                              re.sub(r"[^a-z0-9]+", "", idioma_destino.lower())[:3])
    stem = ruta_entrada.stem
    # Si el stem ya tiene un código de idioma al final (.ja, .en, etc.), lo quitamos.
    m = re.match(r"^(.*)\.([a-z]{2,3})$", stem)
    if m:
        stem = m.group(1)
    return ruta_entrada.with_name(f"{stem}.{codigo}.srt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Traduce y mejora un archivo .srt usando Ollama.",
    )
    p.add_argument("srt", help="Ruta al archivo .srt de entrada.")
    p.add_argument(
        "--output", "-o",
        help="Ruta del .srt de salida. Por defecto: <entrada>.<codigo_idioma>.srt",
    )
    p.add_argument("--idioma-origen", "-s", default="Japanese",
                   help="Idioma de origen (en inglés). Default: Japanese.")
    p.add_argument("--idioma-destino", "-t", default="English",
                   help="Idioma de destino (en inglés). Default: English.")
    p.add_argument("--modelo", "-m", default="vanilj/mistral-nemo-12b-celeste-v1.9",
                   help="Modelo de Ollama. Default: vanilj/mistral-nemo-12b-celeste-v1.9  "
                        "Otros sugeridos: qwen2.5:7b, qwen2.5:32b.")
    p.add_argument("--prompt-file",
                   help="Archivo de texto con un prompt de sistema personalizado. "
                        "Puede contener {idioma_origen} y {idioma_destino}.")
    p.add_argument("--chunk", type=int, default=15,
                   help="Líneas por bloque enviado al modelo. Default: 15. "
                        "Bájalo si ves timeouts; súbelo si quieres más velocidad.")
    p.add_argument("--temperatura", type=float, default=0.5,
                   help="Temperatura del modelo. Default: 0.3 (más fiel).")
    p.add_argument("--num-ctx", type=int, default=16384,
                   help="Tamaño de contexto. Default: 16384.")
    p.add_argument("--timeout", type=float, default=600.0,
                   help="Timeout por llamada HTTP a Ollama (segundos). Default: 600.")
    p.add_argument("--reintentos", type=int, default=2,
                   help="Reintentos por chunk fallido. Default: 2.")
    p.add_argument("--host", default=None,
                   help="URL del servidor de Ollama (ej. http://localhost:11434). "
                        "Por defecto se usa el del cliente.")
    p.add_argument("--max-chars-por-linea", type=int, default=50,
                   help="Máximo de caracteres por línea visible. Default: 42 "
                        "(estándar de subtitulado). Usa 0 para desactivar el ajuste.")
    p.add_argument("--max-lineas", type=int, default=2,
                   help="Máximo de líneas por subtítulo antes de dividir en varios. "
                        "Default: 2.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ruta_in = Path(args.srt).expanduser().resolve()
    if not ruta_in.exists():
        print(f"❌ No existe el archivo: {ruta_in}", file=sys.stderr)
        return 1
    if ruta_in.suffix.lower() != ".srt":
        print(f"⚠️  La entrada no tiene extensión .srt ({ruta_in.suffix}). "
              "Intentaré parsearlo igualmente.")

    ruta_out = construir_ruta_salida(ruta_in, args.idioma_destino, args.output)
    if ruta_out.resolve() == ruta_in.resolve():
        print("❌ La ruta de salida es igual a la de entrada. "
              "Usa --output para especificar otra.", file=sys.stderr)
        return 1

    prompt = cargar_prompt(args)

    try:
        mejorar_srt(
            ruta_entrada=ruta_in,
            ruta_salida=ruta_out,
            idioma_origen=args.idioma_origen,
            idioma_destino=args.idioma_destino,
            modelo=args.modelo,
            prompt_sistema=prompt,
            chunk_size=args.chunk,
            temperatura=args.temperatura,
            num_ctx=args.num_ctx,
            timeout=args.timeout,
            reintentos=args.reintentos,
            host=args.host,
            max_chars_por_linea=args.max_chars_por_linea,
            max_lineas=args.max_lineas,
        )
    except FileNotFoundError as e:
        print(f"❌ Archivo no encontrado: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  Interrumpido por el usuario.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"❌ Error inesperado: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
# Despliegue en RunPod Serverless — guía paso a paso

## Archivos del proyecto

```
proyecto/
├── Dockerfile
├── requirements-runpod.txt
├── mejorar_subtitulos.py    ← tu script existente (sin cambios)
├── runpod_handler.py        ← el handler de RunPod
└── cliente_runpod.py        ← cliente local que llama al endpoint
```

---

## 1. Construir y subir la imagen Docker

Necesitas una cuenta en Docker Hub (o cualquier registry público accesible desde RunPod).

```bash
# Desde la carpeta con todos los archivos
docker build -t TU_USUARIO_DOCKERHUB/subtitle-translator:v1 .
docker push TU_USUARIO_DOCKERHUB/subtitle-translator:v1
```

**Si no tienes Docker local**, puedes construir directamente en RunPod usando "GitHub integration" — apuntando a un repo público que contenga estos archivos. RunPod construye la imagen por ti.

---

## 2. Crear un Network Volume (recomendado, opcional)

Si **no** pre-descargas el modelo dentro de la imagen, cada vez que un worker arranque "fresco" tendrá que descargar qwen2.5:14b (~9 GB). Con un Network Volume, el modelo persiste entre arranques.

1. En RunPod → **Storage** → **New Network Volume**
2. Tamaño: **20 GB** suficiente para qwen2.5:14b. Más si vas a probar varios modelos.
3. Región: elige la misma donde vas a desplegar el endpoint
4. Anótala — la usarás en el siguiente paso

El handler ya está preparado: si detecta `/runpod-volume`, guarda y lee modelos desde ahí.

---

## 3. Crear el endpoint serverless

En RunPod → **Serverless** → **New Endpoint**

| Campo | Valor |
|---|---|
| Endpoint Name | `subtitle-translator` |
| Docker Image | `TU_USUARIO_DOCKERHUB/subtitle-translator:v1` |
| Container Disk | 20 GB (mínimo para CUDA + Ollama) |
| GPU types | RTX 4000 Ada o A4000 (24GB VRAM, baratas) para qwen2.5:14b. Si usas 7b, una RTX 3070/A2000 basta. |
| Active Workers | **0** (claves del modelo serverless: solo arranca con request) |
| Max Workers | 1-3 según uso esperado |
| Idle Timeout | 5 segundos |
| Execution Timeout | 1800s (30 min) para SRTs largos |
| Network Volume | (del paso 2 si lo creaste) montado en `/runpod-volume` |

Después de crear el endpoint, RunPod te muestra un **Endpoint ID** (ej: `ab1cd2efghij34`). Guárdalo.

---

## 4. Generar tu API key

RunPod → **Settings** → **API Keys** → **Create API Key** (permiso "Read & Write")

---

## 5. Probar el endpoint

```bash
# Configura las variables en tu PC
# Windows PowerShell:
$env:RUNPOD_API_KEY = "tu_api_key"
$env:RUNPOD_ENDPOINT_ID = "ab1cd2efghij34"

# Linux/Mac:
export RUNPOD_API_KEY="tu_api_key"
export RUNPOD_ENDPOINT_ID="ab1cd2efghij34"

# Lanzar traducción
python cliente_runpod.py "video.srt" --async
```

### Sync vs Async

- **`--async`** (recomendado para tu caso): manda el job y hace polling. Soporta jobs largos sin timeouts intermedios. **El primer request siempre será async** porque el cold start (descarga del modelo + carga en VRAM) puede tomar 1-5 minutos.
- **sync** (default): bloquea la conexión HTTP. Solo úsalo para SRTs cortos o cuando el worker ya esté warm.

---

## 6. Llamarlo como API desde tu app/script

```python
import requests

resp = requests.post(
    "https://api.runpod.ai/v2/AB1CD2EFGHIJ34/runsync",
    headers={"Authorization": "Bearer TU_API_KEY"},
    json={
        "input": {
            "srt": open("video.srt", encoding="utf-8").read(),
            "idioma_origen": "Japanese",
            "idioma_destino": "English",
            "modelo": "qwen2.5:14b",
        }
    },
    timeout=600,
)
data = resp.json()
print(data["output"]["srt"])
```

O con curl:

```bash
curl -X POST https://api.runpod.ai/v2/AB1CD2EFGHIJ34/run \
  -H "Authorization: Bearer TU_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"srt":"...","idioma_destino":"English"}}'
```

---

## Costos estimados

Con RTX 4000 Ada (~$0.39/hora en RunPod):

| Escenario | Tiempo de worker | Costo aprox. |
|---|---|---|
| SRT pequeño (200 líneas), worker warm | ~60s | $0.007 |
| SRT pequeño, cold start (modelo en volumen) | ~120s | $0.013 |
| SRT pequeño, cold start (descarga modelo) | ~5 min | $0.033 |
| SRT grande (1500 líneas, video 2h), warm | ~5-8 min | $0.04-0.05 |

Pagas **solo segundos de ejecución**. Con `Active Workers = 0` y `Idle Timeout = 5s`, no pagas nada cuando no hay requests.

---

## Tips y gotchas

- **Cold start**: la primera invocación tras inactividad descarga el modelo (si no está en volumen) y lo carga en VRAM. Esperate 2-5 min. Las siguientes en la misma "ventana caliente" son rápidas.
- **Mantener warm**: si necesitas latencia baja, pon `Active Workers = 1`. Pagas la GPU 24/7 pero respuestas instantáneas.
- **Logs**: RunPod → tu endpoint → pestaña "Logs" o "Requests". Ahí ves los `print(...)` del handler.
- **Probar localmente antes**: el handler de RunPod tiene modo dev. Lanza:
  ```bash
  python runpod_handler.py --rp_serve_api
  ```
  Y prueba con:
  ```bash
  curl -X POST http://localhost:8000/runsync -H "Content-Type: application/json" \
    -d '{"input":{"srt":"...","idioma_destino":"English"}}'
  ```
  Esto sirve para depurar sin gastar GPU en RunPod.
- **Archivos grandes**: si tus SRT pasan de ~1MB, mejor subirlos a S3/Cloud Storage y pasar `srt_url` en vez de `srt` inline.
- **Errores frecuentes**:
  - `worker failed to start` → revisa logs del build. Usualmente es CUDA/cuDNN incompatible. La imagen base recomendada arriba ya es compatible con Ollama.
  - `model not found` → el `pull` falló o no terminó. Mira logs.
  - timeouts en `/runsync` → cambia a `--async`.

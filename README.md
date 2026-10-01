# HuevoPS Backend

## Detector

El backend carga el modelo YOLO local `app/modelo_nuevo/best (1).pt` al iniciar.
Las predicciones de imagen (`/detect`) y video (`/video`) usan sus clases `Sano` y `Roto`.
La ruta se resuelve desde el archivo de la aplicacion, independientemente del directorio de arranque.
Si el modelo no se puede cargar, el arranque falla para evitar predicciones con otro detector.
No se requiere configurar Roboflow.

## Direcciones e IP

La IP que figura en la documentación existente es **44.199.34.125**. Confirma
que siga asignada a tu servidor antes de usarla; no se ha comprobado su
accesibilidad pública. Si cambia, sustituye la IP en el frontend y los ejemplos.

| Uso | URL |
| --- | --- |
| API en el servidor documentado | `http://44.199.34.125:8000` |
| WebSocket en el servidor documentado | `ws://44.199.34.125:8000/video` |
| API local | `http://localhost:8000` |
| Swagger UI | `http://44.199.34.125:8000/docs` |
| ReDoc | `http://44.199.34.125:8000/redoc` |
| Esquema OpenAPI | `http://44.199.34.125:8000/openapi.json` |

`0.0.0.0` es la dirección de escucha de Uvicorn en todas las interfaces;
para conectarte utiliza la IP o el dominio del servidor. `localhost` solo sirve
si el cliente se ejecuta en el mismo equipo que el backend.

## Endpoints

Todas las rutas parten de la URL base de la API.

| Método | Ruta | Entrada | Respuesta |
| --- | --- | --- | --- |
| GET | `/health` | Sin parámetros | 200, `{"status":"ok"}` |
| POST | `/detect` | Imagen en campo multipart `file` | 200, JPEG anotado y cabecera `X-Detection-Count` |
| WebSocket | `/video` | JSON inicial `start`, seguido de bytes WebM | Mensajes JSON de detecciones |
| POST | `/train` | ZIP en `file`, opcionales `epochs`, `imgsz`, `batch` | 202, identificador y estado del trabajo |
| GET | `/train/{job_id}` | Identificador devuelto como `jobId` | Estado, parámetros y progreso disponible |
| GET | `/train/{job_id}/model` | Identificador del trabajo completado | Descarga del modelo `.pt` |
| GET | `/train/{job_id}/log` | Identificador del trabajo | Registro en texto plano |
| GET | `/docs` | Sin parámetros | Documentación interactiva HTTP |
| GET | `/redoc` | Sin parámetros | Documentación HTTP |
| GET | `/openapi.json` | Sin parámetros | Esquema de la API HTTP |

Errores previstos: `/detect` devuelve 415 si el tipo no es imagen y 400 si
no puede decodificarla. `/train` devuelve 413 si el ZIP supera el límite,
422 si el dataset o los parámetros no son válidos y 409 si ya hay un
entrenamiento activo. Las consultas de trabajos inexistentes devuelven 404;
la descarga del modelo devuelve 409 hasta que el trabajo esté completado.
Los campos obligatorios ausentes producen errores de validación 422.

## Requisitos y estructura

El despliegue descrito usa Linux/Ubuntu, Python con `venv` y `pip`, y systemd
para mantener el proceso activo. El entorno local inspeccionado utiliza Python
3.14.4; `requirements.txt` no fija una versión de Python ni versiones exactas
de todas las dependencias. La instalación debe resolver correctamente PyTorch
y Ultralytics para la versión y arquitectura del servidor.

```text
Backend/
├── app/main.py                  # API e inferencia de imágenes/vídeo
├── app/training.py              # Validación de datasets y trabajos
├── app/train_worker.py          # Entrenamiento en un proceso separado
├── app/modelo_nuevo/best (1).pt  # Modelo obligatorio al arrancar
├── requirements.txt
├── huevops-backend.service
├── tests/
└── training_jobs/               # Se crea al recibir entrenamientos
```

No requiere base de datos ni credenciales de Roboflow. El vídeo utiliza FFmpeg
a través de `imageio-ffmpeg`. El entrenamiento está configurado para CPU y
necesita permisos de escritura en el repositorio para crear `training_jobs/`.
Reserva espacio para los ZIP, datasets extraídos y resultados: se conservan
entre reinicios y no existe limpieza automática.

## Instalación y arranque manual

Copia o clona el repositorio en el servidor, incluyendo el modelo `.pt`.
Los siguientes comandos asumen la ubicación `/home/ubuntu/Backend`:

```bash
cd /home/ubuntu/Backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Comprueba que existe el modelo obligatorio.
test -f 'app/modelo_nuevo/best (1).pt'
# Comprueba que FFmpeg está disponible para el vídeo.
python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Usa **un solo worker**: el control de entrenamiento simultáneo se mantiene en
memoria dentro de cada proceso. No uses `--reload` en el servicio desplegado.

En otra terminal, verifica el arranque:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/detect \
  -F 'file=@foto.jpg' -D cabeceras.txt -o resultado.jpg
```

El primer comando debe devolver `{"status":"ok"}`; el segundo guarda la imagen
anotada y sus cabeceras. Sustituye `foto.jpg` por una imagen local.

### Configuración de entorno

`LOG_LEVEL` configura el nivel de logs (por defecto `INFO`):

```bash
LOG_LEVEL=DEBUG .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

El servicio systemd carga opcionalmente `/home/ubuntu/Backend/.env`, donde
puedes definir `LOG_LEVEL=INFO`. El arranque manual mostrado no carga `.env`
automáticamente. La IP de escucha y el puerto se configuran en el comando
Uvicorn; la ruta del modelo y los parámetros de vídeo están definidos en
`app/main.py`. No publiques secretos ni credenciales en el README.

## Despliegue persistente con systemd

El archivo `huevops-backend.service` incluido utiliza estas rutas absolutas:

- `WorkingDirectory=/home/ubuntu/Backend`
- `ExecStart=/home/ubuntu/Backend/.venv/bin/uvicorn ...`
- `EnvironmentFile=-/home/ubuntu/Backend/.env` (opcional)

Si instalas en otra ubicación, adapta esas tres rutas y el enlace del ejemplo.
Detén el arranque manual antes de activar el servicio para liberar el puerto.
Ejecuta como el usuario propietario del proyecto:

```bash
mkdir -p ~/.config/systemd/user
ln -sf /home/ubuntu/Backend/huevops-backend.service ~/.config/systemd/user/huevops-backend.service
systemctl --user daemon-reload
systemctl --user enable --now huevops-backend.service
sudo loginctl enable-linger "$USER"
systemctl --user status huevops-backend.service
curl --fail http://127.0.0.1:8000/health
```

`enable-linger` permite mantener el servicio sin una sesión abierta e iniciarlo
al arrancar el servidor. El servicio también reinicia el proceso si falla.

```bash
# Ver logs en directo.
journalctl --user -u huevops-backend.service -f
# Aplicar cambios de código o del modelo.
systemctl --user restart huevops-backend.service
# Detener el backend.
systemctl --user stop huevops-backend.service
```

Si modificas el archivo `.service`, ejecuta `systemctl --user daemon-reload`
antes de reiniciarlo. Un reinicio corta las sesiones de vídeo y el entrenamiento
activo; los trabajos no se reanudan automáticamente.

### Red e integración del frontend

Para acceso directo, permite TCP 8000 en el firewall del servidor y en las
reglas de red del proveedor, restringiéndolo a los clientes necesarios.
Comprueba desde el equipo cliente `http://IP_DEL_SERVIDOR:8000/health`.

Para un frontend publicado con HTTPS, configura un dominio y un proxy inverso
con TLS que reenvíe a `127.0.0.1:8000`, incluyendo las conexiones WebSocket a
`/video`. Las URL serán `https://tu-dominio.com` y
`wss://tu-dominio.com/video`. Ajusta el límite de subida del proxy para permitir
el ZIP de hasta 512 MiB más el contenedor multipart, y sus tiempos de espera
para las subidas y sesiones de vídeo. El repo no incluye configuración del
proxy ni certificados TLS.

La aplicación actual no implementa autenticación ni middleware CORS. Para
publicarla, configura el control de acceso en el proxy o la red, especialmente
para `/train`. Para peticiones HTTP desde otro origen, configura los orígenes
permitidos y expón `X-Detection-Count` si el frontend necesita leer esa cabecera,
o sirve frontend y API bajo el mismo origen.

## Protocolo de vídeo

`/video` usa WebSocket nativo. Al abrir la conexión envía un mensaje de texto:

```json
{
  "type": "start",
  "sessionId": "sesion-001",
  "mimeType": "video/webm;codecs=vp8",
  "width": 1280,
  "height": 720
}
```

Después envía los fragmentos binarios WebM del mismo `MediaRecorder`, en orden,
incluyendo el primer fragmento con su cabecera. No los conviertas a Base64.
`sessionId` debe ser una cadena no vacía y las dimensiones, enteros positivos.
El backend usa las dimensiones reales del vídeo al decodificarlo.

Cada respuesta contiene las detecciones; `box` representa
`[x, y, ancho, alto]` normalizados respecto al fotograma completo, y
`confidence` está entre 0 y 1. `label` mantiene el valor `Huevo`, mientras
`status` distingue `Sano` y `Roto`. Cierra el WebSocket al detener la captura.
Un inicio inválido cierra la conexión con código 1008; un formato no WebM,
con 1003; un fallo de inferencia, con 1011.

Consulta [INTEGRACION_FRONTEND.txt](INTEGRACION_FRONTEND.txt) para los detalles
del cliente. Swagger documenta las rutas HTTP; el protocolo WebSocket se
describe aquí.

## Prueba desde Bruno

Crear una peticion `POST http://localhost:8000/detect` con `Body > Multipart Form`:

- Campo: `file`
- Tipo: `File`
- Valor: una imagen de huevos

La respuesta es `image/jpeg` con las cajas dibujadas. El encabezado `X-Detection-Count`
indica cuantas detecciones se dibujaron.

Cada caja muestra visualmente `Sano` o `Roto` sobre la imagen. El campo JSON
`label` permanece como `Huevo` para mantener el contrato del frontend. El estado y la confianza proceden del modelo YOLO.

El endpoint `/health` devuelve `{"status":"ok"}`.
## Video: rendimiento

La camara puede transmitir a 30 FPS. El backend analiza como maximo 5 fotogramas
por segundo y toma el mas reciente disponible, con `imgsz=416`. `/detect` mantiene
`imgsz=640`. El numero real de predicciones depende de la carga y de las sesiones
simultaneas. Los logs muestran la duracion de inferencia cada 30 respuestas.
Las cajas siguen normalizadas respecto al fotograma completo.

Medicion local con imagen sintetica de 1280x720, seis inferencias calientes:
640: mediana 390 ms; 480: 231 ms; 416: 180 ms. Servidor de 2 CPU, sin CUDA.
La reduccion de resolucion debe validarse con huevos y grietas reales.

## Entrenar con un dataset YOLO

`POST /train` recibe multipart/form-data:

- `file`: ZIP de hasta 512 MB (hasta 2 GB extraidos).
- `epochs`: 1–100, por defecto 20.
- `imgsz`: 320–640, multiplo de 32, por defecto 416.
- `batch`: 1–8, por defecto 2.

Estructura del ZIP (puede estar dentro de una carpeta):

```text
data.yaml
images/train/foto1.jpg
images/val/foto2.jpg
labels/train/foto1.txt
labels/val/foto2.txt
```

Tambien se acepta la estructura Roboflow `train/images`, `train/labels`,
`valid/images`, `valid/labels`, con `data.yaml` en la carpeta que las contiene.
Se requieren conjuntos de entrenamiento y validacion separados, ambos con cajas.
Usa fotos distintas en cada conjunto para evaluar el resultado de forma fiable.

Contenido de `data.yaml`:

```yaml
names: [Sano, Roto]
```

Cada linea del TXT corresponde a una caja:

```text
0 0.5 0.5 0.4 0.6
```

Formato: `clase centro_x centro_y ancho alto`, coordenadas normalizadas entre
0 y 1. Clase `0=Sano`, clase `1=Roto`. Cada imagen debe tener un TXT con el mismo
nombre; un TXT vacio representa una imagen de fondo sin huevos.
El backend genera su propia configuracion de rutas y no ejecuta directivas del YAML subido.

```bash
curl -X POST http://44.199.34.125:8000/train \
  -F 'file=@dataset.zip' \
  -F 'epochs=20' \
  -F 'imgsz=416' \
  -F 'batch=2'
```

Devuelve HTTP 202 con `jobId`, `status`, parametros y conteos del dataset.
Consulta `GET /train/{jobId}` para ver `running`, `completed`, `failed` o
`interrupted`. El campo `progress` aparece tras la primera epoca.
Si el servicio se detiene, el trabajo se interrumpe; no se reanuda automaticamente.

- `GET /train/{jobId}/model`: descarga `best.pt` cuando termina (409 mientras no este listo).
- `GET /train/{jobId}/log`: consulta el registro del entrenamiento.
- `GET /docs`: documentacion interactiva y formulario para probar los endpoints.

Solo se admite un entrenamiento simultaneo (409 para otro trabajo).
Los datasets, logs y resultados se conservan en `Backend/training_jobs/{jobId}`.
El entrenamiento afina una copia del modelo actual en un proceso de CPU con
prioridad reducida. Puede ralentizar las predicciones: preferiblemente entrenar
fuera de las sesiones de video. El modelo resultante se descarga para evaluarlo;
no sustituye automaticamente al modelo que sirve `/detect` y `/video`.

## Comprobaciones y problemas frecuentes

- **No arranca:** revisa los logs y la presencia de `app/modelo_nuevo/best (1).pt`.
  La aplicación falla al iniciar si el modelo no se puede cargar.
- **Puerto ocupado:** detén el proceso manual o el servicio duplicado que escucha
  en 8000 antes de iniciar otra instancia.
- **Funciona localmente pero no desde otro equipo:** revisa la IP, la escucha
  en `0.0.0.0` y las reglas de firewall/red.
- **Vídeo sin resultados:** comprueba FFmpeg, el mensaje `start` y que envías
  un flujo WebM continuo con su primer fragmento.
- **Entrenamiento fallido:** consulta `/train/{job_id}/log`, el espacio en disco
  y la estructura del dataset.

Para ejecutar las pruebas existentes desde la raíz del repo:

```bash
.venv/bin/python -m unittest discover -s tests
```

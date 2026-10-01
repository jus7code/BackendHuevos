# HuevoPS Backend

## Detector

El backend carga el modelo YOLO local `app/modelo_nuevo/best (1).pt` al iniciar.
Las predicciones de imagen (`/detect`) y video (`/video`) usan sus clases `Sano` y `Roto`.
La ruta se resuelve desde el archivo de la aplicacion, independientemente del directorio de arranque.
Si el modelo no se puede cargar, el arranque falla para evitar predicciones con otro detector.
No se requiere configurar Roboflow.

## Instalacion

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Arranque

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Para dejar el backend ejecutandose en segundo plano, iniciarlo automaticamente
al arrancar el sistema y reiniciarlo si el proceso falla:

```bash
mkdir -p ~/.config/systemd/user
ln -sf /home/ubuntu/Backend/huevops-backend.service ~/.config/systemd/user/huevops-backend.service
systemctl --user daemon-reload
systemctl --user enable --now huevops-backend.service
loginctl enable-linger "$USER"
```

Comandos utiles:

```bash
systemctl --user status huevops-backend.service
systemctl --user restart huevops-backend.service
journalctl --user -u huevops-backend.service -f
```

El frontend puede conectarse a `ws://localhost:8000/video`.

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

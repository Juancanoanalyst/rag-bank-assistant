# Asistente RAG sobre el sitio web de BBVA Colombia

Asistente conversacional que responde preguntas sobre el contenido público de
<https://www.bbva.com.co/>, sin depender de búsquedas manuales. Extrae el sitio con web
scraping, lo indexa en una base de datos vectorial y responde citando las páginas de origen.
Recuerda la conversación por ID de sesión y ofrece una pestaña de analítica sobre el histórico.

Todo el stack es gratuito y auto-hospedado, y se levanta con un solo comando.

```bash
docker compose up --build
```

Después abre <http://localhost:8501>.

## Contenido

1. [Cómo funciona](#cómo-funciona)
2. [Requisitos previos](#requisitos-previos)
3. [Puesta en marcha paso a paso](#puesta-en-marcha-paso-a-paso)
4. [Cómo usar la interfaz](#cómo-usar-la-interfaz)
5. [API](#api)
6. [Configuración](#configuración)
7. [Patrones de diseño](#patrones-de-diseño)
8. [Stack tecnológico y justificación](#stack-tecnológico-y-justificación)
9. [Estructura del proyecto](#estructura-del-proyecto)
10. [Pruebas](#pruebas)
11. [Resultados medidos](#resultados-medidos)
12. [Limitaciones conocidas](#limitaciones-conocidas)
13. [Supuestos](#supuestos)
14. [Futuras mejoras](#futuras-mejoras)
15. [Desarrollo asistido por IA](#desarrollo-asistido-por-ia)

## Cómo funciona

```
INGESTA (una vez, servicio "ingest")
  sitemap.xml + robots.txt ──> crawler ──> data/raw/*.html        (datos crudos)
                                              │ trafilatura
                                              v
                                data/clean/documents.jsonl         (datos limpios)
                                              │ splitter recursivo
                                              v
                              chunks ──> embeddings ──> Qdrant

CONSULTA (cada pregunta, POST /chat)
  pregunta + últimos N mensajes de la sesión
        │ 1. condensar: el LLM reescribe la pregunta para que se entienda sola
        v
  búsqueda vectorial en Qdrant (TOP_K candidatos)
        │ 2. reranker: un cross-encoder se queda con los RERANK_TOP_N mejores
        v
  ¿mejor puntaje >= MIN_RERANK_SCORE? ── no ──> "no encontré información"
        │ sí
        v
  3. el LLM responde solo con ese contexto ──> respuesta + fuentes
        │
        v
  4. se guarda el intercambio con sus métricas en SQLite
```

## Requisitos previos

- **Docker** con Docker Compose v2 (probado con Docker Desktop 29.8 en Windows 11 con WSL 2).
- **Memoria para Docker: 8 GB recomendados.** Con 4 GB funciona usando un modelo más pequeño
  (ver [Limitaciones conocidas](#limitaciones-conocidas)).
- **Unos 10 GB de disco** para imágenes, modelos y datos.
- **Conexión a internet en el primer arranque**: se descargan las imágenes, el LLM (~2 GB), el
  reranker (~1,1 GB), el modelo de embeddings (~0,2 GB) y las páginas del sitio.
- **Variables de entorno: ninguna es obligatoria.** Los valores por defecto funcionan tal cual.
  `GROQ_API_KEY` solo hace falta si se elige el proveedor Groq.

No se necesita Python instalado: todo corre en contenedores.

## Puesta en marcha paso a paso

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/Juancanoanalyst/rag-bank-assistant.git
   cd rag-bank-assistant
   ```

2. (Opcional) Crear el archivo de configuración para cambiar algún parámetro:

   ```bash
   cp .env.example .env
   ```

   Si no existe `.env`, se usan los valores por defecto.

3. Levantar el sistema:

   ```bash
   docker compose up --build
   ```

4. Esperar a que termine el primer arranque. Compose ejecuta, en orden:

   | Servicio | Qué hace |
   |---|---|
   | `qdrant` | Base de datos vectorial |
   | `ollama` | Servidor del LLM |
   | `ollama-init` | Descarga el modelo `OLLAMA_MODEL` (una sola vez) |
   | `ingest` | Rastrea el sitio, limpia el HTML e indexa (una sola vez) |
   | `api` | Arranca cuando los dos anteriores terminan bien |
   | `ui` | Arranca cuando la API está sana |

   La primera vez tarda varios minutos: en la máquina de prueba, unos 6 minutos después de
   descargar las imágenes (3 de rastreo con 1 s de pausa entre páginas, 2 de indexación y la
   descarga del modelo). Los arranques siguientes tardan segundos, porque `ingest` detecta que
   el contenido ya está indexado y no repite nada.

5. Abrir <http://localhost:8501>.

Comandos útiles:

```bash
docker compose stop                 # detener sin perder datos
docker compose down -v              # borrar todo, incluidos modelos e historial
docker compose logs -f api          # ver los logs de la API
docker compose run --rm ingest python -m rag_assistant.ingest --force   # volver a rastrear e indexar
```

## Cómo usar la interfaz

La interfaz tiene una barra lateral y dos pestañas.

**Barra lateral — sesión.** Al entrar se genera un ID de sesión. El historial se guarda por ese
ID: para retomar una conversación anterior basta con escribir el mismo ID; "Nueva sesión"
genera uno nuevo.

**Pestaña Chat.** Escribe una pregunta en el campo inferior. La respuesta muestra:

- el texto, generado solo a partir del contenido del sitio;
- las **fuentes**: las páginas de donde salió, en un desplegable;
- el tiempo de respuesta.

Las preguntas de seguimiento funcionan sin repetir el tema. Por ejemplo, después de preguntar
"¿Qué necesito para solicitar un crédito de vivienda?", la pregunta "¿y qué documentos piden?"
se entiende como referida al crédito de vivienda.

Si el sitio no contiene la respuesta, el asistente contesta "No encontré información sobre
eso…" en lugar de inventar.

**Pestaña Analítica.** Recorre todo el histórico de conversaciones y muestra:

- uso: sesiones, preguntas y mensajes por sesión;
- calidad y rendimiento: latencia p50 y p95, porcentaje de preguntas sin respuesta y puntaje
  medio del reranker;
- impacto: horas ahorradas estimadas = preguntas respondidas × `MANUAL_SEARCH_MINUTES`;
- las páginas más recuperadas y los mensajes de cada sesión.

## API

La interfaz usa una API REST disponible en <http://localhost:8000> (documentación interactiva
en `/docs`).

| Método y ruta | Descripción |
|---|---|
| `POST /chat` | Cuerpo `{"session_id": "...", "question": "..."}`. Devuelve respuesta, fuentes, puntajes, pregunta reescrita y latencia |
| `GET /sessions/{id}/history` | Mensajes guardados de una sesión |
| `GET /metrics` | Métricas agregadas del histórico |
| `GET /health` | Chequeo de salud |

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "question": "¿Cómo funciona Bre-B?"}'
```

Errores: entrada inválida → 422; LLM no disponible → 502; buscador o modelos no disponibles →
503; el resto → 500. La respuesta lleva un mensaje apto para mostrar; el detalle técnico queda
en los logs.

## Configuración

Toda la configuración está externalizada en variables de entorno o en `.env`
([.env.example](.env.example) las documenta todas). Las principales:

| Variable | Por defecto | Para qué sirve |
|---|---|---|
| `HISTORY_MAX_MESSAGES` | `6` | **N mensajes** previos de la sesión que se usan como contexto |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Modelo del LLM local |
| `LLM_PROVIDER` | `ollama` | `ollama` o `groq` (este último requiere `GROQ_API_KEY`) |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Modelo de embeddings |
| `RERANKER_MODEL` | `jinaai/jina-reranker-v2-base-multilingual` | Modelo del reranker |
| `RERANKER_PROVIDER` | `fastembed` | `none` desactiva el reranker |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `450` / `70` | Tamaño y solapamiento de los chunks, en caracteres |
| `TOP_K` | `20` | Candidatos que trae la búsqueda vectorial |
| `RERANK_TOP_N` | `5` | Chunks que llegan al LLM |
| `MIN_RERANK_SCORE` | `0.15` | Umbral por debajo del cual se responde "no encontré información" |
| `SCRAPER_MAX_PAGES` | `150` | Páginas a descargar |
| `SCRAPER_BASE_URL` | `https://www.bbva.com.co` | Sitio a rastrear |
| `MANUAL_SEARCH_MINUTES` | `5` | Minutos de una búsqueda manual, para estimar el ahorro |

Los valores se validan al arrancar: una configuración inconsistente (por ejemplo, solapamiento
mayor que el tamaño del chunk) detiene el servicio con un mensaje claro. Los valores con opciones
fijas distinguen mayúsculas: `LOG_LEVEL=INFO`, no `info`.

Después de cambiar `EMBEDDING_MODEL`, `CHUNK_SIZE` o `SCRAPER_MAX_PAGES` hay que volver a indexar
con el comando `--force` de arriba.

## Patrones de diseño

Se usan cuatro patrones, uno por cada punto donde el sistema necesita poder cambiar.

### Strategy — modelos intercambiables

- **Dónde:** [`embeddings/base.py`](src/rag_assistant/embeddings/base.py) (`Embedder`),
  [`reranking/base.py`](src/rag_assistant/reranking/base.py) (`Reranker`) y
  [`llm/base.py`](src/rag_assistant/llm/base.py) (`LLMClient`). Implementaciones:
  `FastEmbedEmbedder`; `FastEmbedReranker` y `NoOpReranker`; `OllamaLLM` y `GroqLLM`.
- **Por qué:** el modelo es la decisión que más probablemente cambie (otro LLM, otro proveedor,
  desactivar el reranker). El resto del código depende solo de la interfaz, así que el cambio es
  una variable de entorno. También permite que las pruebas usen dobles sin descargar modelos.

### Factory — construir desde la configuración

- **Dónde:** [`factory.py`](src/rag_assistant/factory.py) (`build_embedder`, `build_reranker`,
  `build_llm`, `build_retriever`, `build_rag_service`…).
- **Por qué:** es el único módulo que sabe qué clase concreta corresponde a cada valor de
  configuración. Añadir un proveedor es registrar una entrada, sin tocar quien lo usa.

### Facade — una sola puerta de entrada

- **Dónde:** [`service/rag_service.py`](src/rag_assistant/service/rag_service.py) (`RAGService`).
- **Por qué:** responder una pregunta implica historial, condensación, búsqueda, reranking,
  generación y persistencia. `ask(session_id, question)` esconde esa secuencia, de modo que la
  API es una capa delgada y la lógica se prueba sin HTTP.

### Repository — acceso al historial

- **Dónde:** [`history/repository.py`](src/rag_assistant/history/repository.py)
  (`HistoryRepository`, `SQLiteHistoryRepository`).
- **Por qué:** el servicio y la analítica leen y escriben mensajes sin ver SQL. Cambiar SQLite
  por PostgreSQL sería escribir otra implementación de la misma interfaz.

El almacén vectorial sigue la misma idea con
[`vectorstore/base.py`](src/rag_assistant/vectorstore/base.py) (`VectorStore`).

## Stack tecnológico y justificación

| Pieza | Elección | Por qué |
|---|---|---|
| Scraping | `requests` + `trafilatura` | El sitio entrega HTML completo sin JavaScript, así que no hace falta un navegador. `trafilatura` extrae el texto principal y descarta menús y pies |
| `robots.txt` | `protego` | El parser de la librería estándar ignora reglas con comodín, y el sitio usa `Disallow: *.content.html` |
| Chunking | Splitter recursivo propio | Unas cien líneas; evita traer LangChain por una sola clase |
| Embeddings y reranker | `fastembed` (ONNX) | Corre en CPU sin PyTorch: imagen más pequeña y arranque más rápido. Modelos multilingües porque el contenido está en español |
| Base vectorial | Qdrant | Auto-hospedada, un solo contenedor, y con modo en memoria que permite probar sin servidor |
| LLM | Ollama con `qwen2.5:3b` | Local y gratuito, con buen español para su tamaño. Groq queda como alternativa opcional más rápida |
| Historial | SQLite | Cero servicios extra; suficiente para el volumen de una herramienta interna |
| API | FastAPI | Validación con Pydantic y documentación automática |
| Interfaz | Streamlit | Chat funcional en pocas líneas |
| Configuración | `pydantic-settings` | Tipos y validación al arrancar |

**Decisiones medidas, no supuestas:**

- **Tamaño de chunk 450.** El modelo de embeddings ignora lo que pase de 128 tokens (se
  comprobó que el texto posterior no cambia el vector). Con 800 caracteres se perdía la cola de
  cada chunk.
- **Modelo de embeddings.** `jinaai/jina-embeddings-v2-base-es` lee 512 tokens y acertó 3 de 3
  consultas de prueba frente a 2 de 3, pero es unas 7 veces más lento en CPU (1,1 frente a 8,2
  chunks por segundo). Se dejó el rápido por defecto pensando en el arranque, y el otro
  documentado como opción en `.env.example`.
- **Reranker.** Separa bien lo que es del banco de lo que no: las preguntas del sitio obtienen
  puntajes de 0,25 a 0,74 y las ajenas de 0,04 a 0,06. De ahí sale el umbral de 0,15.

## Estructura del proyecto

```
src/rag_assistant/
  config.py            configuración (pydantic-settings)
  factory.py           Factory: construye los componentes
  models.py            estructuras de datos compartidas
  exceptions.py        errores de dominio
  ingest.py            entrada del servicio "ingest": rastrear + limpiar + indexar
  scraping/            crawler, robots, sitemap, limpieza, almacenamiento local
  chunking/            splitter recursivo
  embeddings/          Strategy: Embedder
  vectorstore/         VectorStore y su implementación con Qdrant
  indexing/            pipeline de indexación
  reranking/           Strategy: Reranker
  retrieval/           búsqueda en dos etapas
  llm/                 Strategy: LLMClient (Ollama, Groq)
  history/             Repository: historial en SQLite
  service/             Facade: RAGService y prompts
  analytics/           métricas sobre el histórico
  api/                 FastAPI
  ui/                  Streamlit
tests/                 pruebas con pytest y fixtures HTML
data/raw, data/clean   datos crudos y limpios (se generan, no se versionan)
Dockerfile, docker-compose.yml
```

## Pruebas

```bash
pip install -e ".[dev]"
pytest --cov=rag_assistant
ruff check . && ruff format --check .
```

- 305 pruebas; cobertura del 92 % (la página de Streamlit se ejercita con `AppTest`, que la
  herramienta de cobertura no contabiliza).
- **No necesitan red, modelos ni servicios**: el HTTP se simula con `responses`, los modelos con
  dobles, Qdrant corre en memoria y el scraper usa fixtures HTML guardados en `tests/fixtures`.
- Requieren Python 3.12.
- Un flujo de GitHub Actions ([.github/workflows/ci.yml](.github/workflows/ci.yml)) ejecuta el
  linter y las pruebas en cada push y pull request.

## Resultados medidos

Corrida real de punta a punta con `docker compose up --build` (Windows 11, Docker Desktop,
6 CPU, 4 GB de RAM para Docker, modelo `qwen2.5:1.5b`):

- **Ingesta:** 150 páginas descargadas de las 1.240 del sitemap, 149 documentos limpios (una
  descartada por tener menos de 200 caracteres), 1.122 chunks.
- **Preguntas reales:**

  | Pregunta | Resultado |
  |---|---|
  | ¿Qué necesito para solicitar un crédito de vivienda? | Respondida con el artículo de crédito de vivienda (puntaje 0,65) |
  | ¿y qué documentos piden? | Reescrita como "¿Qué documentos necesitas para solicitar un crédito de vivienda?" y respondida |
  | ¿Cómo funciona Bre-B para recibir dinero? | Respondida, con dos fuentes |
  | ¿Cuál es la capital de Francia? | "No encontré información" (puntaje 0,06), sin llamar al LLM |

- **Latencia en esa máquina:** entre 29 y 45 s por pregunta respondida; 148 s la primera, que
  incluyó descargar y cargar el reranker; 7 s la pregunta sin respuesta.

## Limitaciones conocidas

**Recursos y rendimiento**

- **El modelo por defecto `qwen2.5:3b` no se ha ejecutado.** La máquina de prueba solo da 4 GB a
  Docker y el modelo no cabe junto al reranker, así que la verificación se hizo con
  `qwen2.5:1.5b` (`OLLAMA_MODEL=qwen2.5:1.5b` en `.env`). Con 4 GB hay que hacer lo mismo.
- **La latencia en CPU es alta** (decenas de segundos). El reranker aporta unos 4 s por pregunta
  y el resto es el LLM. Se reduce con más CPU y RAM, bajando `TOP_K`, con
  `RERANKER_PROVIDER=none` o con `LLM_PROVIDER=groq`.
- **El proveedor Groq solo se ha probado contra respuestas simuladas**, nunca contra el servicio
  real. No reintenta ante límites de uso (HTTP 429).

**Cobertura del contenido**

- **Se indexan 150 de las 1.240 páginas** por defecto, muestreadas a intervalos regulares del
  sitemap para cubrir todas las secciones en proporción. Muchas preguntas concretas caen en
  páginas que no están; subir `SCRAPER_MAX_PAGES` lo mejora a costa de un arranque más largo.
- **Solo HTML.** No se procesan los PDF (tarifas, reglamentos), donde suele estar el detalle de
  tasas y costos.
- **No hay actualización incremental.** El contenido es una foto del momento de la ingesta;
  refrescarlo es volver a indexar con `--force`.
- Cerca del 4 % de los chunks supera los 128 tokens del modelo de embeddings, porque el título
  de la página se antepone y no cuenta en `CHUNK_SIZE`; su cola no influye en la búsqueda.

**Calidad de las respuestas**

- Un modelo de 1,5–3 mil millones de parámetros puede resumir de más o mezclar datos de chunks
  distintos. Mitigaciones: solo responde con el contexto recuperado, se muestran las fuentes y
  hay un umbral de relevancia. No hay una evaluación sistemática de calidad.
- La detección de "sin respuesta" depende de un umbral calibrado con pocas preguntas de prueba.

**Operación y seguridad**

- **Sin autenticación ni límite de peticiones.** Cualquiera con acceso a la red puede usar la
  API y leer el historial de una sesión si conoce su ID. Es aceptable para una demo local, no
  para producción.
- **Las preguntas que fallan no se guardan**, así que los errores no aparecen en la analítica.
- `/metrics` lee todo el historial en cada llamada; habría que agregarlo en SQL si creciera.
- Un cambio de modelo de embeddings con el mismo tamaño de vector no se detecta
  automáticamente: hay que reindexar con `--force`.
- Las páginas guardadas a mano sin URL reconocible se citan como `local://archivo`.

**Licencias**

- `jina-reranker-v2-base-multilingual` (CC BY-NC 4.0) y `qwen2.5:3b` (licencia de investigación
  de Qwen) no permiten uso comercial. Sirven para esta prueba; un uso productivo exigiría
  cambiarlos, lo que es un ajuste de configuración gracias al patrón Strategy.

**Atajos tomados**

- Los fixtures HTML de las pruebas son páginas sintéticas escritas a mano con la estructura del
  sitio real, para no depender de la red ni redistribuir contenido del banco.
- No hay migraciones de esquema para SQLite: la tabla se crea si no existe.

## Supuestos

- **Sitio objetivo.** Se usó bbva.com.co. Su `robots.txt` permite el rastreo (`Allow: /`, con
  dos exclusiones que se respetan). El scraper se identifica con un user-agent propio, espera
  1 s entre peticiones y no incluye ninguna técnica para evadir protecciones: si el sitio lo
  rechaza, se detiene. Como respaldo, puede ingerir páginas guardadas a mano en `data/raw`.
- **Usuarios internos en una red de confianza**, por eso no hay autenticación.
- **Idioma.** Preguntas y contenido en español; los modelos y los prompts se eligieron para eso.
- **"N mensajes anteriores"** cuenta mensajes, no intercambios: `HISTORY_MAX_MESSAGES=6` son
  tres pares de pregunta y respuesta. Los intercambios que terminaron en "sin respuesta" no se
  reenvían al modelo.
- **"Sin respuesta"** significa que el mejor puntaje del reranker no llega a `MIN_RERANK_SCORE`
  o que el propio modelo declara que el contexto no alcanza.
- **Horas ahorradas** = preguntas respondidas × `MANUAL_SEARCH_MINUTES` / 60. Es una estimación
  con un parámetro configurable, no una medición.
- **Columna adicional.** La tabla `messages` tiene, además de las columnas pedidas, `answered`,
  necesaria para calcular la tasa de preguntas sin respuesta.
- **Datos crudos y limpios en local** se interpretó como `data/raw` (HTML más un manifiesto) y
  `data/clean/documents.jsonl`, dentro de un volumen de Docker.

## Futuras mejoras

- **Evaluación de calidad**: un conjunto de preguntas con respuesta esperada y métricas de
  recuperación y fidelidad, para comparar modelos, tamaños de chunk y umbrales con datos.
- **Búsqueda híbrida** (vectorial + palabras clave) para nombres de producto y cifras exactas.
- **Ingesta de PDF** y actualización incremental por fecha de modificación del sitemap.
- **Respuestas en streaming** para que la espera se perciba menor.
- **Autenticación, límite de peticiones** y registro de las preguntas fallidas.
- **Agregación de métricas en SQL** y paginación del historial.
- **Persistir la pregunta reescrita** para diagnosticar fallos de recuperación.
- **Pruebas de integración en CI** con Qdrant como servicio.
- **Soporte de GPU** para Ollama cuando esté disponible.

## Desarrollo asistido por IA

Este proyecto se desarrolló con asistencia de IA, usando **Claude Code** (modelo Claude Opus
de Anthropic), y lo declaro abiertamente:

- **Forma de trabajo.** El desarrollo se hizo por fases, cada una en su rama y su pull request.
  Yo definí el alcance, el stack y el orden, revisé cada entrega y aprobé cada fusión; la IA
  escribió el código, las pruebas y esta documentación bajo esa supervisión.
- **Revisión.** En varias fases se usaron agentes de IA independientes para revisar el código y
  buscar casos sin probar. Sus hallazgos y las correcciones quedaron registrados en los pull
  requests #2, #3 y #5.
- **Verificación.** Lo que aquí se afirma que funciona fue ejecutado: las pruebas
  automatizadas, el rastreo real del sitio y la corrida completa con Docker. Lo que no se pudo
  verificar está listado en [Limitaciones conocidas](#limitaciones-conocidas).
- **Decisiones.** Las decisiones técnicas se tomaron con mediciones propias de este proyecto
  (límite de tokens del modelo de embeddings, velocidad de los modelos, puntajes del reranker)
  y están explicadas en este documento y en los mensajes de commit.

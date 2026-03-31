# Docker Setup

## Build

```bash
docker build -t orca-catalogue .
```

## Run locally

```bash
# Prod (catalogue only)
docker run -p 8501:8501 orca-catalogue

# Dev (all pages)
docker run -p 8501:8501 -e APP_ENV=dev orca-catalogue
```

Open `http://localhost:8501`

## Deploy to registry

```bash
# Build with registry tag
docker build -t us-central1-docker.pkg.dev/okapi-274503/esp-docker/orca-catalogue:latest .

# Push
docker push us-central1-docker.pkg.dev/okapi-274503/esp-docker/orca-catalogue:latest
```

Or retag an existing local build:
```bash
docker tag orca-catalogue:latest us-central1-docker.pkg.dev/okapi-274503/esp-docker/orca-catalogue:latest
docker push us-central1-docker.pkg.dev/okapi-274503/esp-docker/orca-catalogue:latest
```

## Notes

- Authenticate once: `gcloud auth configure-docker us-central1-docker.pkg.dev`
- Cloud Run expects port `8080` by default — update `--server.port` in the Dockerfile CMD if needed
- `APP_ENV=prod` is baked into the image; override at runtime with `-e APP_ENV=dev`

# Multi-stage Dockerfile for Unified SatQuery AI Deployment
FROM node:20-alpine AS frontend-builder
WORKDIR /app
COPY package*.json tsconfig.json tailwind.config.ts postcss.config.mjs next.config.mjs ./
COPY src ./src
RUN npm install && npm run build

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS runner
WORKDIR /app

# System dependencies for GDAL, OpenCV, Rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    nodejs \
    npm \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=frontend-builder /app/.next /app/.next
COPY --from=frontend-builder /app/node_modules /app/node_modules

EXPOSE 8000 3001

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & npm run start"]

# Tunisie Telecom Retention Analytics — Frontend
# Multi-stage build: compile the React app with Node, then serve the static
# production build with nginx (never the Vite dev server in production).

FROM node:20-alpine AS build
WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .

# Baked in at build time — the app calls this base path for all API
# requests. Left as "/api" by default so nginx below can reverse-proxy it
# to the backend container and the browser never needs to know the
# backend's host/port (no CORS, no hardcoded ports in the bundle).
ARG VITE_API_BASE_URL=/api
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

## Target 127.0.0.1 explicitly, not "localhost" — this minimal image resolves
## "localhost" to the IPv6 loopback first, and nginx below only binds IPv4,
## which made the healthcheck fail with "connection refused" despite the
## server being up (verified while testing this exact image).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=5 \
    CMD wget -q -O- http://127.0.0.1:80/ >/dev/null || exit 1

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

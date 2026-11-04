FROM node:26-bookworm-slim AS build
# better-sqlite3 is a native module: if no prebuilt binary matches the platform
# it compiles from source, which needs a toolchain. That toolchain stays in this
# stage; the runtime image only receives the compiled result.
RUN apt-get update && apt-get install -y --no-install-recommends python3 make g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY tsconfig.json ./
COPY src ./src
RUN npm run build && npm prune --omit=dev

FROM node:26-bookworm-slim
ENV NODE_ENV=production
WORKDIR /app
COPY package*.json ./
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
# The only writable path: the SQLite database. Not root.
RUN mkdir -p /data && chown node:node /data
USER node
ENV DATABASE_PATH=/data/app.db
VOLUME /data
EXPOSE 4000
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD node -e "fetch('http://localhost:4000/healthz').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
CMD ["node", "dist/server.js"]

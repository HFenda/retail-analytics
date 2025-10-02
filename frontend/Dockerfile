# -------- build stage --------
FROM node:20-alpine AS build
WORKDIR /app

ARG NEXT_PUBLIC_API_BASE_URL
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL

COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# -------- runtime stage --------
FROM node:20-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production
ENV PORT=3000

# kopiramo lock fajl da npm ci reproducira iste verzije
COPY --from=build /app/package.json /app/package-lock.json ./

# instaliraj prod deps (uklj. next cli)
RUN npm ci --omit=dev

# artefakti builda + statički folder
COPY --from=build /app/.next ./.next

EXPOSE 3000
CMD ["npm","run","start"]

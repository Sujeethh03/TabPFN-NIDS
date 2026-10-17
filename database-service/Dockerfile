
FROM node:22-slim

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY prisma ./prisma
COPY prisma.config.ts ./
COPY src ./src
COPY tsconfig.json ./

ARG DIRECT_URL
ENV DIRECT_URL=${postgresql://user:password@localhost:5432/placeholder}

RUN npm run db:generate

EXPOSE 3000

CMD ["npm", "start"]

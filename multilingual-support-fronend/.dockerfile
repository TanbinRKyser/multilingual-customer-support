FROM node:20 AS build
WORKDIR /ui
COPY multilingual-support-frontend/ .
RUN npm ci && npm run build

FROM nginx:alpine
COPY --from=build /ui/dist /usr/share/nginx/html
EXPOSE 80

# Frontend Dockerfile
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy all frontend files
COPY . .

# Build argument para la URL de la API
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

# Build the React app
RUN npm run build

# Install serve para servir archivos estáticos
RUN npm install -g serve

# Expose port
EXPOSE 3000

# Serve the build folder
CMD ["serve", "-s", "dist", "-l", "3000"]
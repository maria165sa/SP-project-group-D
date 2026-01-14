# Use Node.js 18
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy all frontend files
COPY . .

# Build the React app
RUN npm run build

# Install a simple static server
RUN npm install -g serve

# Expose the port
EXPOSE 3000

# Serve the build folder
CMD ["serve", "-s", "dist", "-l", "3000"]

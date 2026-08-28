import { resolve } from 'path'
import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api/, ''),
      },
    },
  },
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        login: resolve(__dirname, 'src/pages/auth/login.html'),
        signup: resolve(__dirname, 'src/pages/auth/signup.html'),
        dashboard: resolve(__dirname, 'src/pages/dashboard/index.html'),
        scan: resolve(__dirname, 'src/pages/scan/index.html'),
        results: resolve(__dirname, 'src/pages/results/index.html'),
        chatbot: resolve(__dirname, 'src/pages/chatbot/index.html'),
        diet: resolve(__dirname, 'src/pages/diet/index.html'),
        recipes: resolve(__dirname, 'src/pages/recipes/index.html'),
        storage: resolve(__dirname, 'src/pages/storage/index.html'),
      }
    }
  }
})

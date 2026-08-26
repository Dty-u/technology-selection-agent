import { createApp } from 'vue'
import { createPinia } from 'pinia'

import App from './App.vue'
import router from './router'
import '@fontsource-variable/noto-sans-sc'
import '@fontsource-variable/ibm-plex-sans'
import '@fontsource/ibm-plex-mono/400.css'
import '@fontsource/ibm-plex-mono/500.css'
import './assets/styles/tokens.css'
import './assets/styles/base.css'
import './assets/styles/motion.css'

const app = createApp(App)

app.use(createPinia())
app.use(router)

app.mount('#app')

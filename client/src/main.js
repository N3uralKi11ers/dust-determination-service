import { createApp } from 'vue'
import './style.css'
import App from './App.vue'
import axios from './plugins/axios'
// import VueVideoPlayer from '@videojs-player/vue'
// import 'video.js/dist/video-js.css'

const app = createApp(App)

// app use
// app.use(VueVideoPlayer)

app.use(axios, {
	baseUrl: 'http://0.0.0.0:80',
})

app.mount('#app')

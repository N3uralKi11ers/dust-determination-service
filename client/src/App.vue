<template>
	<div>
		<div v-if="videoUploaded">
			<Dropbox @close="videoUploaded = false" />
		</div>
		<div v-else>
			<div class="mt-8">
				<Images :imageId="currentImageNumber" />
				<div class="mt-8 mb-4">
					<Slider @percent="saveCurrentPercent" />
				</div>
				{{ currentPageToTime(Math.floor(currentImageNumber / 2)) }}
			</div>
		</div>
	</div>
</template>

<script>
import axios from 'axios'
import Dropbox from './components/Dropbox.vue'
import Images from './components/Images.vue'
import Slider from './components/Slider.vue'

export default {
	components: {
		Dropbox,
		Images,
		Slider,
	},
	data() {
		return {
			photos: [],
			videoDuration: 0,
			videoUploaded: true,
			currentImagePercent: 0,
			imagesCount: 0,
		}
	},
	// async created() {
	// 	const count = await axios.get('http://0.0.0.0/prediction/count')
	// 	this.imagesCount = count
	// },
	async mounted() {
		const count = await axios.get('http://0.0.0.0/prediction/count')
		this.imagesCount = count.data
	},
	computed: {
		// async imagesCount() {
		// 	const count = await axios.get('http://0.0.0.0/prediction/count')
		// 	return count.data
		// },
		currentImageNumber() {
			return Math.round(this.imagesCount * this.currentImagePercent)
		},
	},
	methods: {
		currentPageToTime(seconds) {
			const minutes = Math.floor(seconds / 60)
			const remainingSeconds = seconds % 60
			const formattedMinutes = String(minutes).padStart(2, '0')
			const formattedSeconds = String(remainingSeconds).padStart(2, '0')
			return `${formattedMinutes}:${formattedSeconds}`
		},
		saveCurrentPercent(percent) {
			this.currentImagePercent = percent
		},
		async getPhotosData() {
			await axios.get('http://0.0.0.0/prediction/').then(resp => {
				const frames = resp.data.frames
				this.photos = frames
			})
		},
		async getFramesCount() {
			await axios.get('http://0.0.0.0/prediction/count').then(resp => {
				const count = resp.data
				this.imagesCount = count
			})
		},
		async getVideoDuration() {
			await axios.get('http://0.0.0.0/prediction/total-time').then(resp => {
				const time = resp.data
				this.videoDuration = time
			})
		},
	},
}
</script>

<style>
#app {
	max-width: 1280px;
	margin: 0 auto;
	padding: 2rem;
	text-align: center;
}
</style>

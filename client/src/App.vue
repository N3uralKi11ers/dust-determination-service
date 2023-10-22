<template>
	<div>
		<div
			v-if="videoUploaded"
			class="items-center justify-center h-screen max-w-xl mx-auto"
		>
			<Dropbox @close="videoUploaded = false" />
		</div>
		<div v-else>
			<Images :imageId="currentImageNumber" />
			<div class="mt-8">
				<Slider @percent="saveCurrentPercent" />
			</div>
			{{ currentImageNumber }}
		</div>
	</div>
</template>

<script>
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
			videoUploaded: true,
			currentImagePercent: 0,
			imagesCount: 17,
		}
	},
	computed: {
		currentImageNumber() {
			return Math.round(this.imagesCount * this.currentImagePercent)
		},
	},
	methods: {
		sendVideo() {},
		saveCurrentPercent(percent) {
			this.currentImagePercent = percent
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

<template>
	<div>
		<div
			v-if="videoUploaded"
			class="items-center justify-center h-screen max-w-xl mx-auto"
		>
			<Dropbox @close="videoUploaded = false" />
		</div>
		<div v-else>
			<Images />
			<div class="mt-8">
				<Slider />
			</div>
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
		}
	},
	methods: {
		sendVideo() {},
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

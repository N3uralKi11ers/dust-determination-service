<template>
	<div>
		<Dropbox />
		<!-- <VideoPlayerView /> -->
	</div>
</template>

<script>
import Dropbox from './components/Dropbox.vue'
// import VideoPlayerView from './components/VideoPlayerView.vue'

export default {
	components: {
		Dropbox,
		// VideoPlayerView,
	},
	data() {
		return {}
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

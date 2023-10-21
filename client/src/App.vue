<template>
	<div>
		<Dropbox />
	</div>
</template>

<script>
import Dropbox from './components/Dropbox.vue'

export default {
	components: {
		Dropbox,
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

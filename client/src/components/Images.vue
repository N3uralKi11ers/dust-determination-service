<template>
	<div>
		<img :src="getImageBefore()" alt="Image_before" />
		<img :src="getImageAfter()" alt="Image_after" />
	</div>
</template>
<script>
export default {
	data() {
		return {
			image_id: 1,
		}
	},
	methods: {
		getImageBefore() {
			return 'http://0.0.0.0/' + 'images/before/' + this.image_id
		},
		getImageAfter() {
			return 'http://0.0.0.0/' + 'images/after/' + this.image_id
		},
	},
}
</script>

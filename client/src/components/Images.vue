<template>
	<div>
		<img
			:src="getImageBefore()"
			alt="Image_before"
			class="inline w-1/2 pr-4 rounded"
		/>
		<img
			:src="getImageAfter()"
			alt="Image_after"
			class="inline w-1/2 pl-4 rounded"
		/>
	</div>
</template>
<script>
export default {
	props: {
		imageId: Number(),
	},
	data() {
		return {}
	},
	methods: {
		getImageBefore() {
			return 'http://0.0.0.0/' + 'images/before/' + this.imageId
		},
		getImageAfter() {
			return 'http://0.0.0.0/' + 'images/after/' + this.imageId
		},
	},
}
</script>

<template>
	<div class="w-full bg-red-50">
		<canvas id="canvas" class="mx-auto w-full h-9"></canvas>
	</div>
</template>
<script setup>
import { ref, onMounted } from 'vue'

function getRelativePosition() {}

onMounted(() => {
	console.log('In onMounted')
	let canvas = document.getElementById('canvas')

	let ctx = canvas.getContext('2d')

	const colors = [
		{ color: 'red', position: 0.1 },
		{ color: 'chartreuse', position: 0.4 },
		{ color: 'red', position: 0.6 },
		{ color: 'chartreuse', position: 0.8 },
		{ color: 'red', position: 1 },
	]

	let gradient = ctx.createLinearGradient(0, 0, canvas.width, 0)

	colors.forEach(function (color) {
		gradient.addColorStop(color.position, color.color)
	})

	ctx.fillStyle = gradient
	ctx.fillRect(0, 0, canvas.width, canvas.height)
})
</script>

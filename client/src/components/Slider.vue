<template>
	<div class="w-full">
		<canvas id="canvas" class="mx-auto w-full h-6 rounded"></canvas>
	</div>
</template>
<script setup>
import { ref, onMounted, onUnmounted, defineEmits, watch } from 'vue'

const emit = defineEmits(['percent'])
const mousePos = ref({ x: 0, y: 0 })
const relativePos = ref(0)

function getMousePos(canvas, evt) {
	let rect = canvas.getBoundingClientRect()
	return {
		x: evt.clientX - rect.left,
		y: evt.clientY - rect.top,
	}
}

function getRelativePos(canvas, pos) {
	return pos.x / canvas.width
}

function updateCanvasWidth(canvas) {
	canvas.width = canvas.parentElement.clientWidth
}

function drawRect(ctx, position) {
	ctx.fillStyle = 'white'
	ctx.fillRect(position.x, 0, 4, 200)
}

function clearCanvas(ctx, canvas) {
	ctx.clearRect(0, 0, canvas.width, canvas.height)
	makeGradient(ctx, canvas)
}

function makeGradient(ctx, canvas) {
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
}

onMounted(() => {
	let canvas = document.getElementById('canvas')
	canvas.width = canvas.parentElement.clientWidth
	window.addEventListener('resize', updateCanvasWidth)

	let ctx = canvas.getContext('2d')

	makeGradient(ctx, canvas)

	document.addEventListener('mousedown', event => {
		mousePos.value = getMousePos(canvas, event)
		relativePos.value = getRelativePos(canvas, mousePos.value)
		console.log(relativePos.value)

		clearCanvas(ctx, canvas)
		drawRect(ctx, mousePos.value)
	})
})

onUnmounted(() => {
	window.removeEventListener('resize', updateCanvasWidth)
})

watch(relativePos, pos => {
	emit('percent', pos)
})
</script>

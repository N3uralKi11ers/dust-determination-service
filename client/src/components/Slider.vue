<template>
	<div class="w-full">
		<canvas id="canvas" class="mx-auto w-full h-6 rounded"></canvas>
	</div>
</template>
<script setup>
import axios from 'axios'
import {
	ref,
	onMounted,
	onUnmounted,
	defineProps,
	defineEmits,
	watch,
} from 'vue'

const emit = defineEmits(['percent'])
const mousePos = ref({ x: 0, y: 0 })
const relativePos = ref(0)
const colors = ref([])

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

// async function getPhotosData() {
// 	await axios.get('http://0.0.0.0/prediction/').then(resp => {
// 		const frames = resp.data.frames
// 		this.photos = frames
// 	})
// }

// async function getVideoDuration() {
// 	await axios.get('http://0.0.0.0/prediction/total-time').then(resp => {
// 		const time = resp.data
// 		this.videoDuration = time
// 	})
// },

async function colorsForGradient() {
	const framesResp = await axios.get('http://0.0.0.0/prediction/')
	const frames = framesResp.data.frames

	const time = await axios.get('http://0.0.0.0/prediction/total-time')
	const videoDuration = time.data

	const res = []

	for (const photoData of frames) {
		const percent = photoData.percent
		const time = photoData.time

		res.push({
			color: percent > 0 ? 'red' : 'chartreuse',
			position: time / videoDuration,
		})
	}

	return res
}

function makeGradient(ctx, canvas) {
	let gradient = ctx.createLinearGradient(0, 0, canvas.width, 0)
	colors.value.forEach(function (color) {
		gradient.addColorStop(color.position, color.color)
	})

	ctx.fillStyle = gradient
	ctx.fillRect(0, 0, canvas.width, canvas.height)
}

async function initGradient(ctx, canvas) {
	colors.value = await colorsForGradient()
	let gradient = ctx.createLinearGradient(0, 0, canvas.width, 0)
	colors.value.forEach(function (color) {
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

	initGradient(ctx, canvas)
	// clearCanvas(ctx, canvas)

	document.addEventListener('mousedown', event => {
		mousePos.value = getMousePos(canvas, event)
		relativePos.value = getRelativePos(canvas, mousePos.value)

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

<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { analysisHighlights, gameState, choiceSet } from '$lib/store';
	import { applyChoice, confirmAction } from '$lib/api';
	import {
		BOARD_W, BOARD_H, BUILDING_COORDS, ROAD_COORDS,
		PLAYER_COLORS, PLAYER_COLORS_DIM, INDUSTRY_COLORS, INDUSTRY_ABBR,
		blIdxToCoords, TOWN_BL_RANGES,
		TRADE_POST_TILE_COORDS, TRADE_POST_BEER_COORDS,
		MERCHANT_COLORS, MERCHANT_ABBR,
		coalCubeCoords, ironCubeCoords, COAL_MARKET, IRON_MARKET
	} from '$lib/coords';
	import type { AnalysisHighlights, GameState, Building, Road, TradePostSlot, ChoiceSet as CS } from '$lib/types';

	let canvas: HTMLCanvasElement;
	let boardImg: HTMLImageElement;
	let merchantImgs: Record<string, HTMLImageElement> = {};
	let imagesReady = false;
	let scale = 1;
	let wrapperEl: HTMLElement;
	let rafId: number | null = null;

	$: gs = $gameState;
	$: cs = $choiceSet;
	$: ah = $analysisHighlights;

	$: if (gs && canvas && imagesReady) draw(gs, cs, ah, visualNow(performance.now()));

	onMount(() => {
		const tileTypes = ['all', 'blank', 'cotton', 'goods', 'pottery'];
		let pending = 1 + tileTypes.length;
		const onLoaded = () => { pending--; if (pending === 0) { imagesReady = true; fitBoard(); } };

		boardImg = new Image();
		boardImg.src = '/assets/board.jpg';
		boardImg.onload = onLoaded;

		for (const t of tileTypes) {
			const img = new Image();
			img.src = `/assets/merchants/merchant_${t}.png`;
			img.onload = onLoaded;
			merchantImgs[t] = img;
		}

		window.addEventListener('resize', fitBoard);
		const tick = (t: number) => {
			if (gs && canvas && imagesReady) draw(gs, cs, ah, visualNow(t));
			rafId = requestAnimationFrame(tick);
		};
		const redrawForTestTime = () => {
			if (gs && canvas && imagesReady) draw(gs, cs, ah, visualNow(performance.now()));
		};
		window.addEventListener('brass:advance-time', redrawForTestTime);
		rafId = requestAnimationFrame(tick);
		return () => {
			window.removeEventListener('resize', fitBoard);
			window.removeEventListener('brass:advance-time', redrawForTestTime);
			if (rafId !== null) cancelAnimationFrame(rafId);
		};
	});

	onDestroy(() => {
		if (rafId !== null) cancelAnimationFrame(rafId);
	});

	function fitBoard() {
		if (!wrapperEl || !canvas) return;
		const parent = wrapperEl.parentElement;
		if (!parent) return;
		const maxW = parent.clientWidth - 8;
		const maxH = parent.clientHeight - 8;
		scale = Math.min(maxW / BOARD_W, maxH / BOARD_H);
		if (scale <= 0) scale = 0.5;
		canvas.width = BOARD_W * scale;
		canvas.height = BOARD_H * scale;
		if (gs) draw(gs, cs, ah, visualNow(performance.now()));
	}

	function visualNow(fallback: number): number {
		return (window as typeof window & { __brassVisualTime?: number }).__brassVisualTime ?? fallback;
	}

	function draw(state: GameState, choices: CS | null, highlights: AnalysisHighlights | null, nowMs: number) {
		const ctx = canvas.getContext('2d')!;
		const s = scale;
		ctx.clearRect(0, 0, canvas.width, canvas.height);
		if (boardImg?.complete) ctx.drawImage(boardImg, 0, 0, BOARD_W * s, BOARD_H * s);

		// Draw roads
		for (const road of state.roads) {
			const c = ROAD_COORDS[road.index];
			if (!c) continue;
			ctx.beginPath();
			ctx.arc(c[0]*s, c[1]*s, 10*s, 0, Math.PI*2);
			ctx.fillStyle = PLAYER_COLORS[road.owner] || '#888';
			ctx.fill();
			ctx.strokeStyle = '#000';
			ctx.lineWidth = 2*s;
			ctx.stroke();
		}

		// Draw merchant tiles
		for (const tp of state.trade_posts) {
			drawMerchantTile(ctx, s, tp);
		}

		// Draw merchant beer
		for (const tp of state.trade_posts) {
			drawMerchantBeer(ctx, s, tp);
		}

		// Draw buildings
		for (const b of state.buildings) {
			drawBuilding(ctx, s, b);
		}

		// Draw coal market
		drawCoalMarket(ctx, s, state.coal_market);
		// Draw iron market
		drawIronMarket(ctx, s, state.iron_market);

		drawAnalysisHighlights(ctx, s, highlights, nowMs);

		// Draw top-layer selection UI so every selectable option stays visible.
		drawBoardSelectionOverlay(ctx, s, choices, nowMs);
	}

	function drawAnalysisHighlights(
		ctx: CanvasRenderingContext2D,
		s: number,
		highlights: AnalysisHighlights | null,
		nowMs: number
	) {
		if (!highlights) return;
		const pulse = 0.7 + 0.3 * Math.sin(nowMs / 320);
		ctx.save();
		ctx.lineWidth = (3 + pulse) * s;
		ctx.strokeStyle = `rgba(8, 127, 91, ${0.78 + pulse * 0.18})`;
		for (const location of highlights.building_locations) {
			const co = blIdxToCoords(location);
			if (!co) continue;
			ctx.strokeRect((co[0] - 31) * s, (co[1] - 31) * s, 62 * s, 62 * s);
		}

		ctx.setLineDash([7 * s, 5 * s]);
		ctx.strokeStyle = `rgba(0, 105, 148, ${0.72 + pulse * 0.2})`;
		for (const location of highlights.source_locations) {
			const co = blIdxToCoords(location);
			if (!co) continue;
			ctx.beginPath();
			ctx.arc(co[0] * s, co[1] * s, 34 * s, 0, Math.PI * 2);
			ctx.stroke();
		}
		ctx.setLineDash([]);

		ctx.strokeStyle = `rgba(181, 64, 49, ${0.8 + pulse * 0.16})`;
		for (const road of highlights.roads) {
			const co = ROAD_COORDS[road];
			if (!co) continue;
			ctx.beginPath();
			ctx.arc(co[0] * s, co[1] * s, 23 * s, 0, Math.PI * 2);
			ctx.stroke();
		}

		for (const slot of highlights.merchant_slots) {
			const co = TRADE_POST_TILE_COORDS[slot];
			if (!co) continue;
			ctx.strokeRect((co[0] - 4) * s, (co[1] - 4) * s, 56 * s, 58 * s);
		}
		ctx.restore();
	}

	type SelectionRect = { x: number; y: number; half: number };
	function getSelectionRects(choices: CS | null): SelectionRect[] {
		if (!choices) return [];
		const out: SelectionRect[] = [];
		if (choices.kind === 'road' || choices.kind === 'second_road') {
			for (const opt of choices.options) {
				const c = ROAD_COORDS[opt.value as number];
				if (!c) continue;
				out.push({ x: c[0], y: c[1], half: 22 });
			}
		}
		if (choices.kind === 'build_location' || choices.kind === 'sell_target') {
			for (const opt of choices.options) {
				const co = blIdxToCoords(opt.value as number);
				if (!co) continue;
				out.push({ x: co[0], y: co[1], half: 34 });
			}
		}
		if (choices.kind === 'beer_source' || choices.kind === 'action_beer_source') {
			for (const opt of choices.options) {
				const co = beerSourceOptionCoords(opt);
				if (!co) continue;
				out.push({ x: co[0], y: co[1], half: 26 });
			}
		}
		return out;
	}

	function drawBoardSelectionOverlay(ctx: CanvasRenderingContext2D, s: number, choices: CS | null, nowMs: number) {
		const rects = getSelectionRects(choices);
		if (rects.length === 0) return;

		// Dim entire board first.
		ctx.save();
		ctx.fillStyle = 'rgba(0,0,0,0.62)';
		ctx.fillRect(0, 0, canvas.width, canvas.height);

		// Punch windows so selectable options are clearly visible.
		ctx.globalCompositeOperation = 'destination-out';
		for (const r of rects) {
			ctx.fillRect((r.x - r.half) * s, (r.y - r.half) * s, r.half * 2 * s, r.half * 2 * s);
		}
		ctx.restore();

		// Pulsating green square borders around selectable options.
		const pulse = 0.55 + 0.45 * Math.sin(nowMs / 250);
		for (const r of rects) {
			const x = (r.x - r.half) * s;
			const y = (r.y - r.half) * s;
			const w = r.half * 2 * s;
			const line = (2.4 + pulse * 2.4) * s;

			ctx.strokeStyle = `rgba(34, 197, 94, ${0.65 + 0.35 * pulse})`;
			ctx.lineWidth = line;
			ctx.strokeRect(x, y, w, w);

			ctx.strokeStyle = `rgba(134, 239, 172, ${0.35 + 0.35 * pulse})`;
			ctx.lineWidth = (1.2 + pulse * 1.2) * s;
			ctx.strokeRect(x - 2 * s, y - 2 * s, w + 4 * s, w + 4 * s);
		}
	}

	function drawCoalMarket(ctx: CanvasRenderingContext2D, s: number, remaining: number) {
		for (let i = 0; i < remaining; i++) {
			const [x, y] = coalCubeCoords(i);
			ctx.fillStyle = '#1a1a1a';
			ctx.fillRect(x * s, y * s, COAL_MARKET.cubeSize * s, COAL_MARKET.cubeSize * s);
			ctx.strokeStyle = '#555';
			ctx.lineWidth = 1 * s;
			ctx.strokeRect(x * s, y * s, COAL_MARKET.cubeSize * s, COAL_MARKET.cubeSize * s);
		}
		ctx.fillStyle = '#000';
		ctx.font = `bold ${14 * s}px sans-serif`;
		ctx.fillText(String(remaining), COAL_MARKET.labelX * s, COAL_MARKET.labelY * s);
	}

	function drawIronMarket(ctx: CanvasRenderingContext2D, s: number, remaining: number) {
		for (let i = 0; i < remaining; i++) {
			const [x, y] = ironCubeCoords(i);
			ctx.fillStyle = '#f97316';
			ctx.fillRect(x * s, y * s, IRON_MARKET.cubeSize * s, IRON_MARKET.cubeSize * s);
			ctx.strokeStyle = '#c05a00';
			ctx.lineWidth = 1 * s;
			ctx.strokeRect(x * s, y * s, IRON_MARKET.cubeSize * s, IRON_MARKET.cubeSize * s);
		}
		ctx.fillStyle = '#f97316';
		ctx.font = `bold ${14 * s}px sans-serif`;
		ctx.fillText(String(remaining), IRON_MARKET.labelX * s, IRON_MARKET.labelY * s);
	}

	function drawBuilding(ctx: CanvasRenderingContext2D, s: number, b: Building) {
		const co = blIdxToCoords(b.location);
		if (!co) return;
		const x = co[0]*s, y = co[1]*s;
		const w = 48*s, h = 48*s;

		ctx.fillStyle = b.flipped ? PLAYER_COLORS_DIM[b.owner] : PLAYER_COLORS[b.owner];
		ctx.fillRect(x-w/2, y-h/2, w, h);
		ctx.strokeStyle = '#000';
		ctx.lineWidth = 2*s;
		ctx.strokeRect(x-w/2, y-h/2, w, h);

		ctx.fillStyle = '#000';
		ctx.fillRect(x-w/2+2*s, y-h/2+2*s, 16*s, 14*s);
		ctx.fillStyle = INDUSTRY_COLORS[b.industry] || '#fff';
		ctx.font = `bold ${10*s}px sans-serif`;
		ctx.fillText(INDUSTRY_ABBR[b.industry] || b.industry.slice(0,2), x-w/2+3*s, y-h/2+13*s);

		ctx.fillStyle = b.flipped ? '#aaa' : '#fff';
		ctx.font = `bold ${12*s}px sans-serif`;
		ctx.textAlign = 'center';
		ctx.fillText('L'+b.level, x, y+4*s);
		ctx.textAlign = 'left';

		if (b.resource_amt > 0) {
			const rc = b.industry === 'Coal' ? '#111' : b.industry === 'Iron' ? '#f97316' : '#d4a054';
			for (let i = 0; i < b.resource_amt; i++) {
				const rx = x - w/2 + 4*s + (i%3)*14*s;
				const ry = y + 10*s + Math.floor(i/3)*14*s;
				if (b.industry === 'Beer') {
					ctx.beginPath(); ctx.arc(rx+5*s,ry+5*s,6*s,0,Math.PI*2);
					ctx.fillStyle = rc; ctx.fill();
				} else {
					ctx.fillStyle = rc; ctx.fillRect(rx,ry,12*s,12*s);
				}
			}
		}
	}

	/** Get [x,y] coords for a beer source option value (beer_source or action_beer_source) */
	function beerSourceOptionCoords(opt: { value: unknown }): [number, number] | null {
		const v = opt.value;
		if (v && typeof v === 'object' && !Array.isArray(v)) {
			const obj = v as Record<string, number>;
			if ('TradePost' in obj) return TRADE_POST_BEER_COORDS[obj.TradePost] ?? null;
			if ('Building' in obj) return blIdxToCoords(obj.Building);
			if ('OwnBrewery' in obj) return blIdxToCoords(obj.OwnBrewery);
			if ('OpponentBrewery' in obj) return blIdxToCoords(obj.OpponentBrewery);
		}
		return null;
	}

	function merchantImageKey(tileType: string | null): string {
		if (!tileType) return 'blank';
		const lower = tileType.toLowerCase();
		if (lower === 'all') return 'all';
		if (lower === 'cotton') return 'cotton';
		if (lower === 'goods') return 'goods';
		if (lower === 'pottery') return 'pottery';
		return 'blank';
	}

	function drawMerchantTile(ctx: CanvasRenderingContext2D, s: number, tp: TradePostSlot) {
		const co = TRADE_POST_TILE_COORDS[tp.slot_index];
		if (!co) return;
		const x = co[0] * s, y = co[1] * s;
		const w = 48 * s, h = 50 * s;

		const key = merchantImageKey(tp.tile_type);
		const img = merchantImgs[key];
		if (img?.complete) {
			ctx.drawImage(img, x, y, w, h);
		} else {
			const color = MERCHANT_COLORS[tp.tile_type ?? 'Blank'] ?? '#555';
			ctx.fillStyle = color;
			ctx.fillRect(x, y, w, h);
			ctx.strokeStyle = '#000';
			ctx.lineWidth = 2 * s;
			ctx.strokeRect(x, y, w, h);
			const abbr = MERCHANT_ABBR[tp.tile_type ?? 'Blank'] ?? '?';
			ctx.fillStyle = '#fff';
			ctx.font = `bold ${14 * s}px sans-serif`;
			ctx.textAlign = 'center';
			ctx.fillText(abbr, x + w / 2, y + h / 2 + 5 * s);
			ctx.textAlign = 'left';
		}
	}

	function drawMerchantBeer(ctx: CanvasRenderingContext2D, s: number, tp: TradePostSlot) {
		if (!tp.has_beer) return;
		const co = TRADE_POST_BEER_COORDS[tp.slot_index];
		if (!co) return;
		const x = co[0] * s, y = co[1] * s;
		const r = 12 * s;
		ctx.beginPath();
		ctx.arc(x, y, r, 0, Math.PI * 2);
		ctx.fillStyle = '#d4a054';
		ctx.fill();
		ctx.strokeStyle = '#8b6914';
		ctx.lineWidth = 2 * s;
		ctx.stroke();
	}

	function handleCanvasClick(e: MouseEvent) {
		if (!cs) return;
		const rect = canvas.getBoundingClientRect();
		const mx = (e.clientX - rect.left) / scale;
		const my = (e.clientY - rect.top) / scale;

		if (cs.kind === 'road' || cs.kind === 'second_road') {
			for (const opt of cs.options) {
				const c = ROAD_COORDS[opt.value as number];
				if (!c) continue;
				if (Math.hypot(mx-c[0], my-c[1]) < 20) {
					applyChoice(cs.kind, opt.value);
					return;
				}
			}
		}
		if (cs.kind === 'build_location' || cs.kind === 'sell_target') {
			for (const opt of cs.options) {
				const co = blIdxToCoords(opt.value as number);
				if (!co) continue;
				if (Math.hypot(mx-co[0], my-co[1]) < 30) {
					applyChoice(cs.kind, opt.value);
					return;
				}
			}
		}
		if (cs.kind === 'beer_source' || cs.kind === 'action_beer_source') {
			for (const opt of cs.options) {
				const co = beerSourceOptionCoords(opt);
				if (!co) continue;
				if (Math.hypot(mx-co[0], my-co[1]) < 22) {
					applyChoice(cs.kind, opt.value);
					return;
				}
			}
		}
	}
</script>

<div class="board-wrapper" bind:this={wrapperEl}>
	<canvas bind:this={canvas} on:click={handleCanvasClick} class="board-canvas"></canvas>
</div>

<style>
	.board-wrapper {
		position: relative;
		width: 100%;
		height: 100%;
		display: flex;
		align-items: center;
		justify-content: center;
	}
	.board-canvas {
		display: block;
		border-radius: 8px;
		box-shadow: 0 4px 20px rgba(0,0,0,0.5);
		cursor: crosshair;
	}
</style>

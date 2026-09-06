<script lang="ts">
	import { onMount } from 'svelte';
	import { Scan, ZoomIn, ZoomOut } from 'lucide-svelte';
	import { allIndustryData, analysisHighlights, cardPreviewTown, gameState, choiceSet } from '$lib/store';
	import { applyChoice, loadIndustryData } from '$lib/api';
	import BoardDetails from './BoardDetails.svelte';
	import {
		boardCamera, clampViewport, zoomViewport, boardChoiceTargets, boardObjectTargets,
		hitBoardTarget, describeBoardTarget, targetKey,
		type BoardViewport, type BoardTarget
	} from '$lib/board-interaction';
	import {
		BOARD_W, BOARD_H, BUILDING_COORDS, ROAD_COORDS,
		PLAYER_COLORS, PLAYER_COLORS_DIM, INDUSTRY_COLORS, INDUSTRY_ABBR,
		blIdxToCoords,
		TRADE_POST_TILE_COORDS, TRADE_POST_BEER_COORDS,
		MERCHANT_COLORS, MERCHANT_ABBR,
		coalCubeCoords, ironCubeCoords, COAL_MARKET, IRON_MARKET
	} from '$lib/coords';
	import type { AnalysisHighlights, GameState, Building, TradePostSlot, ChoiceSet as CS } from '$lib/types';

	export let interactionLocked = false;

	let canvas: HTMLCanvasElement;
	let selectionMask: HTMLCanvasElement;
	let boardImg: HTMLImageElement;
	let merchantImgs: Record<string, HTMLImageElement> = {};
	let imagesReady = false;
	let wrapperEl: HTMLElement;
	let rafId: number | null = null;
	let reducedMotion = false;
	let view: BoardViewport = { width: 1, height: 1, zoom: 1, panX: 0, panY: 0 };
	let hoveredKey: string | null = null;
	let pinnedKey: string | null = null;
	let detailsHeight = 120;
	let keyboardIndex = -1;
	let previousPreview: string | null = null;
	let previousChoices: CS | null = null;
	let dragging = false;
	let applying = false;
	const pointers = new Map<number, { x: number; y: number }>();
	let gesture: { x: number; y: number; panX: number; panY: number; moved: boolean; choices: CS | null } | null = null;
	let pinch: { distance: number; zoom: number; worldX: number; worldY: number } | null = null;

	$: gs = $gameState;
	$: cs = $choiceSet;
	$: ah = $analysisHighlights;
	$: previewTown = $cardPreviewTown;
	$: choices = boardChoiceTargets(interactionLocked ? null : cs);
	$: objects = boardObjectTargets(gs);
	$: targets = [...new Map([...objects, ...choices].map(target => [targetKey(target), target])).values()];
	$: if (pinnedKey && !targets.some(target => targetKey(target) === pinnedKey)) pinnedKey = null;
	$: activeTarget = targets.find(target => targetKey(target) === (pinnedKey ?? hoveredKey)) ?? null;
	$: detail = describeBoardTarget(activeTarget, gs, $allIndustryData);
	$: accessibleTarget = detail ? [detail.title, detail.subtitle, detail.owner?.name, detail.status,
		...detail.stats.map(stat => `${stat.label}: ${stat.value}`)].filter(Boolean).join(', ') : '';
	$: camera = boardCamera(view);
	$: detailWidth = Math.min(252, view.width - 16);
	$: targetX = (activeTarget?.x ?? 0) * camera.scale + camera.x;
	$: targetY = (activeTarget?.y ?? 0) * camera.scale + camera.y;
	$: detailLeft = Math.max(8, Math.min(view.width - detailWidth - 8,
		targetX + detailWidth + 24 < view.width ? targetX + 18 : targetX - detailWidth - 18));
	$: detailTop = Math.max(48, Math.min(view.height - detailsHeight - 8,
		targetY < view.height / 2 ? targetY + 20 : targetY - detailsHeight - 20));
	$: if (previewTown !== previousPreview) {
		previousPreview = previewTown;
		if (previewTown) {
			hoveredKey = null;
			pinnedKey = null;
			keyboardIndex = -1;
			view = revealTown(view, previewTown);
		}
	}
	$: if (cs !== previousChoices) {
		previousChoices = cs;
		hoveredKey = null;
		pinnedKey = null;
		keyboardIndex = -1;
		resetGesture();
	}

	$: if (gs && canvas && imagesReady) draw(gs, cs, ah, previewTown, activeTarget, view, visualNow(performance.now()));

	onMount(() => {
		if (!$allIndustryData) void loadIndustryData();
		selectionMask = document.createElement('canvas');
		const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
		const syncMotion = () => { reducedMotion = motionQuery.matches; };
		syncMotion();
		motionQuery.addEventListener('change', syncMotion);
		const tileTypes = ['all', 'blank', 'cotton', 'goods', 'pottery'];
		let pending = 1 + tileTypes.length;
		const onLoaded = () => { pending--; if (pending === 0) { imagesReady = true; fitBoard(); } };

		boardImg = new Image();
		boardImg.onload = onLoaded;
		boardImg.src = '/assets/board.jpg';

		for (const t of tileTypes) {
			const img = new Image();
			img.onload = onLoaded;
			img.onerror = onLoaded;
			img.src = `/assets/merchants/merchant_${t}.png`;
			merchantImgs[t] = img;
		}

		const resizeObserver = new ResizeObserver(fitBoard);
		resizeObserver.observe(wrapperEl);
		const tick = (t: number) => {
			if (gs && canvas && imagesReady) draw(gs, cs, ah, previewTown, activeTarget, view, visualNow(t));
			rafId = requestAnimationFrame(tick);
		};
		const redrawForTestTime = () => {
			if (gs && canvas && imagesReady) draw(gs, cs, ah, previewTown, activeTarget, view, visualNow(performance.now()));
		};
		window.addEventListener('brass:advance-time', redrawForTestTime);
		rafId = requestAnimationFrame(tick);
		return () => {
			motionQuery.removeEventListener('change', syncMotion);
			resizeObserver.disconnect();
			window.removeEventListener('brass:advance-time', redrawForTestTime);
			if (rafId !== null) cancelAnimationFrame(rafId);
		};
	});

	function fitBoard() {
		if (!wrapperEl || !canvas) return;
		resetGesture();
		view = clampViewport({ ...view, width: Math.max(1, wrapperEl.clientWidth), height: Math.max(1, wrapperEl.clientHeight) });
		canvas.width = view.width;
		canvas.height = view.height;
		if (gs && imagesReady) draw(gs, cs, ah, previewTown, activeTarget, view, visualNow(performance.now()));
	}

	function visualNow(fallback: number): number {
		return (window as typeof window & { __brassVisualTime?: number }).__brassVisualTime ?? fallback;
	}

	function draw(state: GameState, choices: CS | null, highlights: AnalysisHighlights | null, town: string | null, inspected: BoardTarget | null, viewport: BoardViewport, nowMs: number) {
		const ctx = canvas.getContext('2d')!;
		const camera = boardCamera(viewport);
		const s = camera.scale;
		ctx.fillStyle = '#c9cecb';
		ctx.fillRect(0, 0, canvas.width, canvas.height);
		ctx.save();
		ctx.translate(camera.x, camera.y);
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
		ctx.restore();

		// Draw top-layer selection UI so every selectable option stays visible.
		drawBoardSelectionOverlay(ctx, camera, choices, nowMs);
		drawCardPreview(ctx, camera, town, nowMs);
		if (inspected && !town) {
			ctx.save();
			ctx.strokeStyle = '#fff';
			ctx.lineWidth = 2;
			ctx.setLineDash([4, 3]);
			const size = (inspected.half + 3) * s;
			ctx.strokeRect(inspected.x * s + camera.x - size, inspected.y * s + camera.y - size, size * 2, size * 2);
			ctx.restore();
		}
	}

	function drawCardPreview(ctx: CanvasRenderingContext2D, camera: ReturnType<typeof boardCamera>, town: string | null, nowMs: number) {
		const slots = town ? BUILDING_COORDS[town] : null;
		if (!town || !slots?.length) return;
		const s = camera.scale;
		const left = (Math.min(...slots.map(([x]) => x)) - 42) * s + camera.x;
		const top = (Math.min(...slots.map(([, y]) => y)) - 42) * s + camera.y;
		const right = (Math.max(...slots.map(([x]) => x)) + 42) * s + camera.x;
		const bottom = (Math.max(...slots.map(([, y]) => y)) + 42) * s + camera.y;
		const pulse = reducedMotion ? 0.5 : 0.5 + 0.5 * Math.sin(nowMs / 380);

		ctx.save();
		ctx.beginPath();
		ctx.roundRect(left, top, right - left, bottom - top, 8 * s);
		ctx.fillStyle = 'rgba(130, 231, 243, 0.09)';
		ctx.fill();
		ctx.lineWidth = Math.min(7 * s, 6);
		ctx.strokeStyle = 'rgba(14, 34, 38, 0.9)';
		ctx.stroke();
		ctx.lineWidth = Math.min((2.5 + pulse) * s, 3);
		ctx.strokeStyle = '#82e7f3';
		ctx.shadowColor = '#82e7f3';
		ctx.shadowBlur = Math.min((12 + pulse * 6) * s, 18);
		ctx.stroke();
		ctx.shadowBlur = 0;

		// Keep the city name legible in CSS pixels on smaller boards.
		ctx.font = '600 13px Inter, ui-sans-serif, sans-serif';
		const labelWidth = ctx.measureText(town).width + 20;
		const labelX = Math.max(6, Math.min((left + right - labelWidth) / 2, canvas.width - labelWidth - 6));
		let labelY = Math.max(6, top - 28);
		if (labelY < 44 && labelX + labelWidth > canvas.width - 170) labelY = Math.min(canvas.height - 30, bottom + 6);
		ctx.beginPath();
		ctx.roundRect(labelX, labelY, labelWidth, 24, 4);
		ctx.fillStyle = '#142d32';
		ctx.fill();
		ctx.lineWidth = 1;
		ctx.strokeStyle = '#82e7f3';
		ctx.stroke();
		ctx.fillStyle = '#e4fbff';
		ctx.textAlign = 'center';
		ctx.textBaseline = 'middle';
		ctx.fillText(town, labelX + labelWidth / 2, labelY + 12);
		ctx.restore();
	}

	function drawAnalysisHighlights(
		ctx: CanvasRenderingContext2D,
		s: number,
		highlights: AnalysisHighlights | null,
		nowMs: number
	) {
		if (!highlights) return;
		const pulse = reducedMotion ? 0.85 : 0.7 + 0.3 * Math.sin(nowMs / 320);
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

	function drawBoardSelectionOverlay(ctx: CanvasRenderingContext2D, camera: ReturnType<typeof boardCamera>, choices: CS | null, nowMs: number) {
		const rects = boardChoiceTargets(choices);
		if (rects.length === 0) return;
		const s = camera.scale;

		if (selectionMask.width !== canvas.width || selectionMask.height !== canvas.height) {
			selectionMask.width = canvas.width;
			selectionMask.height = canvas.height;
		}
		const maskCtx = selectionMask.getContext('2d')!;
		maskCtx.clearRect(0, 0, canvas.width, canvas.height);
		maskCtx.fillStyle = 'rgba(0,0,0,0.62)';
		maskCtx.fillRect(0, 0, canvas.width, canvas.height);

		// Remove windows from the mask only, preserving the board and pieces below.
		for (const r of rects) {
			maskCtx.clearRect((r.x - r.half) * s + camera.x, (r.y - r.half) * s + camera.y, r.half * 2 * s, r.half * 2 * s);
		}
		ctx.drawImage(selectionMask, 0, 0);

		// Pulsating green square borders around selectable options.
		const pulse = reducedMotion ? 0.75 : 0.55 + 0.45 * Math.sin(nowMs / 250);
		for (const r of rects) {
			const x = (r.x - r.half) * s + camera.x;
			const y = (r.y - r.half) * s + camera.y;
			const w = r.half * 2 * s;
			const line = Math.min((2.4 + pulse * 2.4) * s, 4);

			ctx.strokeStyle = `rgba(34, 197, 94, ${0.65 + 0.35 * pulse})`;
			ctx.lineWidth = line;
			ctx.strokeRect(x, y, w, w);

			ctx.strokeStyle = `rgba(134, 239, 172, ${0.35 + 0.35 * pulse})`;
			ctx.lineWidth = Math.min((1.2 + pulse * 1.2) * s, 2);
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
		if (img?.complete && img.naturalWidth > 0) {
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

	function clearInspection() {
		hoveredKey = null;
		pinnedKey = null;
		keyboardIndex = -1;
	}

	function dismissInspection() {
		const restoreFocus = wrapperEl?.contains(document.activeElement);
		clearInspection();
		if (restoreFocus) wrapperEl.focus({ preventScroll: true });
	}

	function resetGesture() {
		const captured = [...pointers.keys()];
		pointers.clear();
		gesture = null;
		pinch = null;
		dragging = false;
		for (const id of captured) if (canvas?.hasPointerCapture(id)) canvas.releasePointerCapture(id);
	}

	function pointOnBoard(clientX: number, clientY: number) {
		const rect = canvas.getBoundingClientRect();
		const camera = boardCamera(view);
		return { x: (clientX - rect.left - camera.x) / camera.scale, y: (clientY - rect.top - camera.y) / camera.scale };
	}

	function inspectPoint(clientX: number, clientY: number) {
		const point = pointOnBoard(clientX, clientY);
		const hit = hitBoardTarget(choices, point.x, point.y) ?? hitBoardTarget(objects, point.x, point.y);
		hoveredKey = hit ? targetKey(hit) : null;
	}

	function revealedViewport(viewport: BoardViewport, x: number, y: number, half: number): BoardViewport {
		if (viewport.zoom === 1) return viewport;
		const camera = boardCamera(viewport);
		if ((x - half) * camera.scale + camera.x >= 8 && (x + half) * camera.scale + camera.x <= viewport.width - 8 &&
			(y - half) * camera.scale + camera.y >= 44 && (y + half) * camera.scale + camera.y <= viewport.height - 8) return viewport;
		return clampViewport({ ...viewport, panX: (BOARD_W / 2 - x) * camera.scale, panY: (BOARD_H / 2 - y) * camera.scale });
	}

	function revealPoint(x: number, y: number, half = 34) {
		view = revealedViewport(view, x, y, half);
	}

	function revealTown(viewport: BoardViewport, town: string): BoardViewport {
		const slots = BUILDING_COORDS[town];
		if (!slots?.length) return viewport;
		const xs = slots.map(([x]) => x), ys = slots.map(([, y]) => y);
		return revealedViewport(viewport, (Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2,
			Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) / 2 + 42);
	}

	function changeZoom(zoom: number, x = view.width / 2, y = view.height / 2) {
		clearInspection();
		view = zoomViewport(view, zoom, x, y);
	}

	function fitView() {
		resetGesture();
		clearInspection();
		view = { ...view, zoom: 1, panX: 0, panY: 0 };
	}

	function handleWheel(event: WheelEvent) {
		event.preventDefault();
		const rect = canvas.getBoundingClientRect();
		const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? view.height : 1);
		changeZoom(view.zoom * Math.exp(-delta * 0.0015), event.clientX - rect.left, event.clientY - rect.top);
	}

	function startPointer(event: PointerEvent) {
		if (event.button !== 0) return;
		wrapperEl.focus({ preventScroll: true });
		pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
		canvas.setPointerCapture(event.pointerId);
		if (pointers.size === 1) {
			gesture = { x: event.clientX, y: event.clientY, panX: view.panX, panY: view.panY, moved: false, choices: cs };
		} else if (pointers.size === 2) {
			const [a, b] = [...pointers.values()];
			const center = pointOnBoard((a.x + b.x) / 2, (a.y + b.y) / 2);
			pinch = { distance: Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)), zoom: view.zoom, worldX: center.x, worldY: center.y };
			if (gesture) gesture.moved = true;
			clearInspection();
		}
	}

	function movePointer(event: PointerEvent) {
		if (!pointers.has(event.pointerId)) {
			if (event.pointerType !== 'touch') inspectPoint(event.clientX, event.clientY);
			return;
		}
		pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
		if (pointers.size >= 2 && pinch) {
			const [a, b] = [...pointers.values()];
			const rect = canvas.getBoundingClientRect();
			const zoom = Math.max(1, Math.min(4, pinch.zoom * Math.hypot(a.x - b.x, a.y - b.y) / pinch.distance));
			const next = { ...view, zoom, panX: 0, panY: 0 };
			const camera = boardCamera(next);
			view = clampViewport({ ...next,
				panX: (a.x + b.x) / 2 - rect.left - pinch.worldX * camera.scale - camera.x,
				panY: (a.y + b.y) / 2 - rect.top - pinch.worldY * camera.scale - camera.y });
			dragging = true;
		} else if (gesture) {
			const dx = event.clientX - gesture.x, dy = event.clientY - gesture.y;
			if (Math.hypot(dx, dy) > 6) gesture.moved = true;
			if (gesture.moved) {
				clearInspection();
				dragging = true;
				view = clampViewport({ ...view, panX: gesture.panX + dx, panY: gesture.panY + dy });
			}
		}
	}

	function endPointer(event: PointerEvent) {
		if (!pointers.has(event.pointerId)) return;
		const finished = gesture;
		pointers.delete(event.pointerId);
		if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
		pinch = null;
		if (pointers.size) {
			const remaining = [...pointers.values()][0];
			gesture = { ...remaining, panX: view.panX, panY: view.panY, moved: true, choices: cs };
		} else {
			gesture = null;
			dragging = false;
			if (finished && !finished.moved && finished.choices === cs) {
				const point = pointOnBoard(event.clientX, event.clientY);
				void activatePoint(point.x, point.y, event.pointerType === 'touch');
			}
		}
	}

	async function activatePoint(x: number, y: number, touch = false) {
		if (applying) return;
		const hit = hitBoardTarget(choices, x, y, touch ? 12 / boardCamera(view).scale : 0);
		if (hit?.option && cs && !interactionLocked) {
			applying = true;
			clearInspection();
			try { await applyChoice(cs.kind, hit.option.value); }
			finally { applying = false; }
		} else {
			const object = hitBoardTarget(objects, x, y);
			pinnedKey = object ? targetKey(object) : null;
			hoveredKey = null;
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.target !== wrapperEl) return;
		if (event.key === '+' || event.key === '=') { event.preventDefault(); changeZoom(view.zoom * 1.5); }
		else if (event.key === '-') { event.preventDefault(); changeZoom(view.zoom / 1.5); }
		else if (event.key === '0') { event.preventDefault(); fitView(); }
		else if (event.key === 'Escape') dismissInspection();
		else if (event.key === 'Enter' || event.key === ' ') {
			event.preventDefault();
			if (activeTarget) void activatePoint(activeTarget.x, activeTarget.y);
		} else if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) {
			event.preventDefault();
			const available = choices.length ? choices : objects;
			if (!available.length) return;
			const delta = event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? -1 : 1;
			keyboardIndex = event.key === 'Home' ? 0 : event.key === 'End' ? available.length - 1
				: keyboardIndex < 0 ? (delta > 0 ? 0 : available.length - 1)
				: (keyboardIndex + delta + available.length) % available.length;
			const selected = available[keyboardIndex];
			pinnedKey = null;
			hoveredKey = targetKey(selected);
			revealPoint(selected.x, selected.y, selected.half);
		}
	}
</script>

<svelte:window on:blur={() => { resetGesture(); hoveredKey = null; }} on:keydown={event => { if (event.key === 'Escape') dismissInspection(); }} />

<!-- The canvas implements its own arrow-key navigation and target activation. -->
<!-- svelte-ignore a11y-no-noninteractive-tabindex a11y-no-noninteractive-element-interactions -->
<div class="board-wrapper" bind:this={wrapperEl} role="application" tabindex="0" aria-label="Game board" aria-keyshortcuts="ArrowLeft ArrowRight ArrowUp ArrowDown Enter + - 0" on:keydown={handleKeydown}>
	<canvas
		bind:this={canvas} class="board-canvas" aria-hidden="true"
		data-view={JSON.stringify({ zoom: view.zoom, scale: camera.scale, origin_x: camera.x, origin_y: camera.y, width: view.width, height: view.height, inspection: detail?.key ?? null })}
		style:cursor={dragging ? 'grabbing' : applying ? 'wait' : hoveredKey ? 'pointer' : view.zoom > 1 ? 'grab' : 'default'}
		on:pointerdown={startPointer} on:pointermove={movePointer} on:pointerup={endPointer}
		on:pointercancel={resetGesture} on:lostpointercapture={event => { if (pointers.has(event.pointerId)) resetGesture(); }}
		on:pointerleave={() => { if (!pointers.size) hoveredKey = null; }}
		on:wheel|nonpassive={handleWheel}
	></canvas>
	<div class="board-toolbar" role="group" aria-label="Map view">
		<button on:click={() => changeZoom(view.zoom / 1.5)} disabled={view.zoom <= 1} title="Zoom out" aria-label="Zoom out"><ZoomOut size={17} /></button>
		<output aria-label="Map zoom">{Math.round(view.zoom * 100)}%</output>
		<button on:click={() => changeZoom(view.zoom * 1.5)} disabled={view.zoom >= 4} title="Zoom in" aria-label="Zoom in"><ZoomIn size={17} /></button>
		<button on:click={fitView} title="Fit board" aria-label="Fit board"><Scan size={17} /></button>
	</div>
	{#if detail && !previewTown}
		<BoardDetails {detail} pinned={!!pinnedKey} left={detailLeft} top={detailTop} bind:height={detailsHeight} on:close={dismissInspection} />
	{/if}
	<span class="preview-status" role="status">{previewTown ?? accessibleTarget}</span>
</div>

<style>
	.preview-status {
		position: absolute;
		width: 1px;
		height: 1px;
		overflow: hidden;
		clip-path: inset(50%);
		white-space: nowrap;
	}
	.board-wrapper {
		position: relative;
		width: 100%;
		height: 100%;
		overflow: hidden;
	}
	.board-canvas {
		display: block;
		touch-action: none;
		width: 100%;
		height: 100%;
	}
	.board-wrapper:focus-visible { outline: 2px solid #087f5b; outline-offset: -2px; }
	.board-toolbar { position: absolute; top: 8px; right: 8px; z-index: 4; display: flex; align-items: center; border: 1px solid #b2c1b7; border-radius: 5px; background: #f5f8f5; box-shadow: 0 2px 8px #0002; }
	.board-toolbar button { width: 32px; height: 32px; display: grid; place-items: center; border: 0; background: transparent; color: #345844; cursor: pointer; }
	.board-toolbar button:hover:not(:disabled) { background: #dcebe0; }
	.board-toolbar button:disabled { opacity: .35; cursor: default; }
	.board-toolbar button:focus-visible { outline: 2px solid #087f5b; outline-offset: -2px; }
	.board-toolbar output { width: 46px; text-align: center; color: #52675a; font-size: 11px; font-variant-numeric: tabular-nums; }
</style>

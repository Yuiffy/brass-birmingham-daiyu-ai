import {
	BOARD_W, BOARD_H, ROAD_COORDS, ROAD_TOWNS, TOWN_BL_RANGES,
	TRADE_POST_TILE_COORDS, TRADE_POST_BEER_COORDS, blIdxToCoords
} from './coords';
import type { ChoiceOption, ChoiceSet, GameState, IndustryLevelData } from './types';

export interface BoardViewport {
	width: number;
	height: number;
	zoom: number;
	panX: number;
	panY: number;
}

export function boardCamera(view: BoardViewport) {
	const base = Math.max(0.01, Math.min((view.width - 8) / BOARD_W, (view.height - 8) / BOARD_H));
	const scale = base * view.zoom;
	return {
		scale,
		x: (view.width - BOARD_W * scale) / 2 + view.panX,
		y: (view.height - BOARD_H * scale) / 2 + view.panY
	};
}

export function clampViewport(view: BoardViewport): BoardViewport {
	const zoom = Math.max(1, Math.min(4, view.zoom));
	const { scale } = boardCamera({ ...view, zoom });
	const maxX = Math.max(0, (BOARD_W * scale - view.width) / 2);
	const maxY = Math.max(0, (BOARD_H * scale - view.height) / 2);
	return { ...view, zoom, panX: Math.max(-maxX, Math.min(maxX, view.panX)), panY: Math.max(-maxY, Math.min(maxY, view.panY)) };
}

export function zoomViewport(view: BoardViewport, zoom: number, x: number, y: number): BoardViewport {
	const before = boardCamera(view);
	const next = { ...view, zoom: Math.max(1, Math.min(4, zoom)), panX: 0, panY: 0 };
	const after = boardCamera(next);
	next.panX = x - (x - before.x) / before.scale * after.scale - after.x;
	next.panY = y - (y - before.y) / before.scale * after.scale - after.y;
	return clampViewport(next);
}

export interface BoardTarget {
	kind: 'building' | 'road' | 'merchant';
	index: number;
	x: number;
	y: number;
	half: number;
	option?: ChoiceOption;
}

export function targetKey(target: BoardTarget): string {
	return `${target.kind}:${target.index}`;
}

function target(kind: BoardTarget['kind'], index: number, coords: [number, number] | undefined | null, half: number, option?: ChoiceOption): BoardTarget[] {
	return coords ? [{ kind, index, x: coords[0], y: coords[1], half, option }] : [];
}

export function boardChoiceTargets(choices: ChoiceSet | null): BoardTarget[] {
	if (!choices) return [];
	return choices.options.flatMap(option => {
		if (choices.kind === 'road' || choices.kind === 'second_road') return target('road', option.value, ROAD_COORDS[option.value], 22, option);
		if (choices.kind === 'build_location' || choices.kind === 'sell_target') return target('building', option.value, blIdxToCoords(option.value), 34, option);
		if (choices.kind === 'beer_source' || choices.kind === 'action_beer_source') {
			const value = option.value;
			if (!value || typeof value !== 'object' || Array.isArray(value)) return [];
			if ('TradePost' in value) return target('merchant', value.TradePost, TRADE_POST_BEER_COORDS[value.TradePost], 26, option);
			for (const key of ['Building', 'OwnBrewery', 'OpponentBrewery']) {
				if (key in value) return target('building', value[key], blIdxToCoords(value[key]), 26, option);
			}
		}
		return [];
	});
}

export function boardObjectTargets(state: GameState | null): BoardTarget[] {
	if (!state) return [];
	return [
		...state.buildings.flatMap(building => target('building', building.location, blIdxToCoords(building.location), 26)),
		...state.roads.flatMap(road => target('road', road.index, ROAD_COORDS[road.index], 18)),
		...state.trade_posts.flatMap(merchant => {
			const coords = TRADE_POST_TILE_COORDS[merchant.slot_index];
			return target('merchant', merchant.slot_index, coords ? [coords[0] + 24, coords[1] + 25] : null, 25);
		})
	].sort((left, right) => left.y - right.y || left.x - right.x);
}

export function hitBoardTarget(targets: BoardTarget[], x: number, y: number, tolerance = 0): BoardTarget | null {
	let closest: BoardTarget | null = null;
	let distance = Infinity;
	for (const candidate of targets) {
		const half = Math.max(candidate.half, tolerance);
		const dx = x - candidate.x, dy = y - candidate.y;
		const d = dx * dx + dy * dy;
		if (Math.abs(dx) <= half && Math.abs(dy) <= half && d < distance) {
			closest = candidate;
			distance = d;
		}
	}
	return closest;
}

export interface BoardDetail {
	key: string;
	kind: BoardTarget['kind'];
	title: string;
	subtitle: string;
	status: string;
	owner?: { name: string; color: string };
	image?: string;
	stats: { label: string; value: string | number }[];
}

export function describeBoardTarget(
	item: BoardTarget | null,
	state: GameState | null,
	industries: Record<string, IndustryLevelData[]> | null
): BoardDetail | null {
	if (!item || !state) return null;
	const key = targetKey(item);
	if (item.kind === 'building') {
		const building = state.buildings.find(building => building.location === item.index);
		if (!building) {
			if (!item.option) return null;
			const town = Object.entries(TOWN_BL_RANGES).find(([, [start, end]]) => item.index >= start && item.index < end);
			return {
				key, kind: item.kind, title: town?.[0] ?? 'Rural brewery',
				subtitle: town ? `Slot ${item.index - town[1][0] + 1}` : '',
				status: 'Available', stats: []
			};
		}
		const tile = industries?.[building.industry]?.find(tile => tile.level === building.level);
		const stats: BoardDetail['stats'] = [
			{ label: building.flipped ? 'Era VP' : 'Potential VP', value: building.vp_on_flip },
			{ label: 'Link icons', value: building.road_vp }
		];
		if (['Coal', 'Iron', 'Beer'].includes(building.industry)) stats.unshift({ label: building.industry, value: building.resource_amt });
		else if (tile) stats.unshift({ label: 'Beer to sell', value: tile.beer_needed });
		if (tile) stats.push({ label: 'Flip income', value: `${tile.income >= 0 ? '+' : ''}${tile.income}` });
		return {
			key, kind: item.kind, title: `${building.industry} ${building.level}`,
			subtitle: building.town, status: building.flipped ? 'Flipped' : 'Unflipped',
			owner: state.players.find(player => player.index === building.owner),
			image: `/assets/buildings/icons/${building.industry.toLowerCase()}.svg`, stats
		};
	}
	if (item.kind === 'road') {
		const road = state.roads.find(road => road.index === item.index);
		if (!road && !item.option) return null;
		return {
			key, kind: item.kind, title: state.era === 'Railroad' ? 'Rail link' : 'Canal link',
			subtitle: (ROAD_TOWNS[item.index] ?? []).join(' - '), status: road ? 'Built' : 'Available',
			owner: road ? state.players.find(player => player.index === road.owner) : undefined, stats: []
		};
	}
	const merchant = state.trade_posts.find(merchant => merchant.slot_index === item.index);
	if (!merchant) return null;
	const active = merchant.tile_type && merchant.tile_type !== 'Blank';
	return {
		key, kind: item.kind, title: merchant.trade_post === 'Shrewbury' ? 'Shrewsbury' : merchant.trade_post,
		subtitle: 'Merchant', status: active ? 'Active' : 'Closed',
		stats: active ? [
			{ label: 'Accepts', value: merchant.tile_type === 'All' ? 'All goods' : merchant.tile_type! },
			{ label: 'Beer', value: merchant.has_beer ? 'Available' : 'Used' }
		] : []
	};
}

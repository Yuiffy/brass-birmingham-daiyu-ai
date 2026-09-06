/** Board coordinate data from the original pygame render.py. */

import type { Card } from './types';

export const BOARD_W = 1200;
export const BOARD_H = 1200;

export const BUILDING_COORDS: Record<string, [number, number][]> = {
	'Leek': [[632, 92], [687, 92]],
	'Stoke-On-Trent': [[492, 125], [467, 177], [522, 177]],
	'Stone': [[342, 302], [397, 302]],
	'Uttoxeter': [[647, 282], [702, 282]],
	'Belper': [[847, 127], [902, 127], [957, 127]],
	'Derby': [[905, 255], [877, 307], [932, 307]],
	'Stafford': [[452, 412], [507, 412]],
	'Burton-Upon-Trent': [[787, 447], [842, 447]],
	'beer1': [[357, 522]],
	'Cannock': [[537, 532], [592, 532]],
	'Tamworth': [[802, 597], [857, 597]],
	'Walsall': [[607, 672], [662, 672]],
	'Coalbrookdale': [[282, 637], [252, 697], [307, 697]],
	'Wolverhampton': [[417, 642], [472, 642]],
	'Dudley': [[472, 787], [527, 787]],
	'Kidderminster': [[387, 912], [442, 912]],
	'beer2': [[292, 997]],
	'Worcester': [[402, 1062], [457, 1062]],
	'Birmingham': [[722, 777], [777, 777], [722, 832], [777, 832]],
	'Nuneaton': [[912, 712], [967, 712]],
	'Coventry': [[967, 812], [937, 872], [992, 872]],
	'Redditch': [[667, 972], [722, 972]]
};

export const TOWN_BL_RANGES: Record<string, [number, number]> = {
	'Stafford': [0, 2], 'Burton-Upon-Trent': [2, 4], 'Cannock': [4, 6],
	'Tamworth': [6, 8], 'Walsall': [8, 10], 'Leek': [10, 12],
	'Stoke-On-Trent': [12, 15], 'Stone': [15, 17], 'Uttoxeter': [17, 19],
	'Belper': [19, 22], 'Derby': [22, 25], 'Coalbrookdale': [25, 28],
	'Wolverhampton': [28, 30], 'Dudley': [30, 32], 'Kidderminster': [32, 34],
	'Worcester': [34, 36], 'Birmingham': [36, 40], 'Nuneaton': [40, 42],
	'Coventry': [42, 45], 'Redditch': [45, 47]
};

/**
 * Road coordinates indexed by Rust static_data.rs LINK_LOCATIONS road index.
 * Coordinates are pixel positions on the 1200×1200 board image.
 */
export const ROAD_COORDS: [number, number][] = [
	[422, 120],  //  0: Warrington–StokeOnTrent
	[564, 107],  //  1: StokeOnTrent–Leek
	[770, 92],   //  2: Leek–Belper              (rail-only)
	[918, 206],  //  3: Belper–Derby
	[980, 253],  //  4: Derby–Nottingham
	[792, 305],  //  5: Derby–Uttoxeter           (rail-only)
	[899, 401],  //  6: Derby–BurtonUponTrent
	[834, 699],  //  7: Birmingham–Tamworth   (canal+rail)
	[444, 256],  //  8: StokeOnTrent–Stone
	[519, 293],  //  9: Stone–Uttoxeter           (rail-only)
	[383, 402],  // 10: Stone–Stafford
	[622, 359],  // 11: Stone–BurtonUponTrent
	[569, 469],  // 12: Stafford–Cannock
	[686, 477],  // 13: Cannock–BurtonUponTrent   (rail-only)
	[836, 527],  // 14: Tamworth–BurtonUponTrent
	[703, 562],  // 15: Walsall–BurtonUponTrent   (canal-only)
	[462, 520],  // 16: LoneBrewery1–Cannock
	[478, 577],  // 17: Wolverhampton–Cannock
	[645, 597],  // 18: Walsall–Cannock
	[353, 644],  // 19: Wolverhampton–Coalbrookdale
	[203, 644],  // 20: Shrewbury–Coalbrookdale
	[319, 827],  // 21: Kidderminster–Coalbrookdale
	[428, 849],  // 22: Kidderminster–Dudley
	[545, 654],  // 23: Wolverhampton–Walsall
	[450, 730],  // 24: Wolverhampton–Dudley
	[743, 661],  // 25: Tamworth–Walsall          (rail-only)
	[930, 630],  // 26: Tamworth–Nuneaton
	[1025, 780], // 27: Nuneaton–Coventry         (rail-only)
	[663, 759],  // 28: Birmingham–Walsall
	[856, 763],  // 29: Birmingham–Nuneaton       (rail-only)
	[858, 861],  // 30: Birmingham–Coventry
	[856, 916],  // 31: Birmingham–Oxford
	[735, 913],  // 32: Birmingham–Redditch       (rail-only)
	[577, 948],  // 33: Birmingham–Worcester
	[610, 803],  // 34: Birmingham–Dudley
	[797, 994],  // 35: Redditch–Oxford
	[604, 1025], // 36: Redditch–Gloucester
	[526, 1101], // 37: Worcester–Gloucester
	[407, 996],  // 38: Worcester–LoneBrewery2–Kidderminster
];

/** Same link ordering as ROAD_COORDS and Rust LINK_LOCATIONS. */
export const ROAD_TOWNS: string[][] = [
	['Warrington', 'Stoke-On-Trent'], ['Stoke-On-Trent', 'Leek'],
	['Leek', 'Belper'], ['Belper', 'Derby'], ['Derby', 'Nottingham'],
	['Derby', 'Uttoxeter'], ['Derby', 'Burton-Upon-Trent'],
	['Birmingham', 'Tamworth'], ['Stoke-On-Trent', 'Stone'],
	['Stone', 'Uttoxeter'], ['Stone', 'Stafford'], ['Stone', 'Burton-Upon-Trent'],
	['Stafford', 'Cannock'], ['Cannock', 'Burton-Upon-Trent'],
	['Tamworth', 'Burton-Upon-Trent'], ['Walsall', 'Burton-Upon-Trent'],
	['Rural brewery 1', 'Cannock'], ['Wolverhampton', 'Cannock'],
	['Walsall', 'Cannock'], ['Wolverhampton', 'Coalbrookdale'],
	['Shrewsbury', 'Coalbrookdale'], ['Kidderminster', 'Coalbrookdale'],
	['Kidderminster', 'Dudley'], ['Wolverhampton', 'Walsall'],
	['Wolverhampton', 'Dudley'], ['Tamworth', 'Walsall'],
	['Tamworth', 'Nuneaton'], ['Nuneaton', 'Coventry'], ['Birmingham', 'Walsall'],
	['Birmingham', 'Nuneaton'], ['Birmingham', 'Coventry'], ['Birmingham', 'Oxford'],
	['Birmingham', 'Redditch'], ['Birmingham', 'Worcester'], ['Birmingham', 'Dudley'],
	['Redditch', 'Oxford'], ['Redditch', 'Gloucester'], ['Worcester', 'Gloucester'],
	['Worcester', 'Rural brewery 2', 'Kidderminster']
];

/** Trade post tile coordinates: slot_index → [x, y] for the merchant tile image */
export const TRADE_POST_TILE_COORDS: Record<number, [number, number]> = {
	0: [95, 715],     // Shrewbury slot 0
	1: [945, 1020],   // Oxford slot 0
	2: [1000, 1020],  // Oxford slot 1
	3: [680, 1110],   // Gloucester slot 0
	4: [735, 1110],   // Gloucester slot 1
	5: [290, 150],    // Warrington slot 0
	6: [345, 150],    // Warrington slot 1
	7: [1040, 215],   // Nottingham slot 0
	8: [1105, 215],   // Nottingham slot 1
};

/** Beer barrel coordinates for each trade post slot */
export const TRADE_POST_BEER_COORDS: Record<number, [number, number]> = {
	0: [150, 713],    // Shrewbury
	1: [946, 986],    // Oxford slot 0
	2: [1023, 987],   // Oxford slot 1
	3: [682, 1073],   // Gloucester slot 0
	4: [765, 1072],   // Gloucester slot 1
	5: [290, 205],    // Warrington slot 0
	6: [369, 206],    // Warrington slot 1
	7: [1049, 271],   // Nottingham slot 0
	8: [1125, 275],   // Nottingham slot 1
};

/** Coal market: base position and layout for cubes (from render.py drawCoal) */
export const COAL_MARKET = {
	baseX: 1000,
	baseY: 385,
	colOffset: 25,
	rowOffset: 35.5,
	labelX: 1000,
	labelY: 330,
	cubeSize: 15,
};

/** Iron market: base position and layout for cubes (from render.py drawIron) */
export const IRON_MARKET = {
	baseX: 1065,
	baseY: 458,
	colOffset: 25,
	rowOffset: 35.5,
	labelX: 1100,
	labelY: 400,
	cubeSize: 15,
};

/** Get [x,y] for coal cube at index i (0-based) */
export function coalCubeCoords(i: number): [number, number] {
	const x = COAL_MARKET.baseX + (i % 2 === 0 ? 0 : COAL_MARKET.colOffset);
	const y = COAL_MARKET.baseY + Math.floor(i / 2) * COAL_MARKET.rowOffset;
	return [x, y];
}

/** Get [x,y] for iron cube at index i (0-based) */
export function ironCubeCoords(i: number): [number, number] {
	const x = IRON_MARKET.baseX + (i % 2 === 0 ? 0 : IRON_MARKET.colOffset);
	const y = IRON_MARKET.baseY + Math.floor(i / 2) * IRON_MARKET.rowOffset;
	return [x, y];
}

/** Merchant tile type → color for drawing */
export const MERCHANT_COLORS: Record<string, string> = {
	Cotton: '#3b82f6',
	Goods: '#8b5cf6',
	Pottery: '#ec4899',
	All: '#22c55e',
	Blank: '#555',
};

export const MERCHANT_ABBR: Record<string, string> = {
	Cotton: 'Ct',
	Goods: 'Gd',
	Pottery: 'Pt',
	All: '★',
	Blank: '—',
};

export const PLAYER_COLORS = ['#c7a750', '#9c79c6', '#a44529', '#b6c3ca'];
export const PLAYER_COLORS_DIM = ['#7a6430', '#5e4878', '#6b2d1a', '#6e767a'];
export const PLAYER_NAMES = ['Coade', 'Brunel', 'Arkwright', 'Tinsley'];

export const INDUSTRY_COLORS: Record<string, string> = {
	Coal: '#333', Iron: '#f97316', Beer: '#d4a054',
	Goods: '#8b5cf6', Pottery: '#ec4899', Cotton: '#3b82f6'
};
export const INDUSTRY_ABBR: Record<string, string> = {
	Coal: 'Co', Iron: 'Fe', Beer: 'Br',
	Goods: 'Gd', Pottery: 'Pt', Cotton: 'Ct'
};

export function blIdxToCoords(blIdx: number): [number, number] | null {
	if (blIdx === 47) return BUILDING_COORDS['beer1'][0];
	if (blIdx === 48) return BUILDING_COORDS['beer2'][0];
	for (const [town, [start, end]] of Object.entries(TOWN_BL_RANGES)) {
		if (blIdx >= start && blIdx < end) {
			const slotIdx = blIdx - start;
			const coords = BUILDING_COORDS[town];
			if (coords && slotIdx < coords.length) return coords[slotIdx];
		}
	}
	return null;
}

export const TOWN_COLOR_MAP: Record<string, string> = {
	Leek: '#3b82f6', StokeOnTrent: '#3b82f6', Stone: '#3b82f6', Uttoxeter: '#3b82f6',
	Belper: '#22c55e', Derby: '#22c55e',
	Stafford: '#ef4444', BurtonUponTrent: '#ef4444', Cannock: '#ef4444',
	Tamworth: '#ef4444', Walsall: '#ef4444',
	Coalbrookdale: '#eab308', Wolverhampton: '#eab308', Dudley: '#eab308',
	Kidderminster: '#eab308', Worcester: '#eab308',
	Birmingham: '#a855f7', Nuneaton: '#a855f7', Coventry: '#a855f7', Redditch: '#a855f7',
};

export function isLocationCard(cardType: string): boolean {
	return cardType.startsWith('Location');
}

const CARD_TOWN_ALIASES: Record<string, string> = {
	StokeOnTrent: 'Stoke-On-Trent',
	BurtonUponTrent: 'Burton-Upon-Trent'
};

export function locationCardTown(card: Card): string | null {
	if (!isLocationCard(card.card_type)) return null;
	const town = CARD_TOWN_ALIASES[card.label] ?? card.label;
	return Object.prototype.hasOwnProperty.call(TOWN_BL_RANGES, town) ? town : null;
}

export function townCardColor(label: string): string {
	return TOWN_COLOR_MAP[label] ?? '#888';
}

/** Map a card label to a card image filename (industry cards only). */
export function cardImage(label: string, cardType: string): string {
	if (cardType === 'WildLocation') return '/assets/cards/wild_location.png';
	if (cardType === 'WildIndustry') return '/assets/cards/wild_industry.png';
	const lower = label.toLowerCase();
	if (lower.includes('beer') || lower.includes('brewery')) return '/assets/cards/brewery.png';
	if (lower.includes('coal')) return '/assets/cards/coal_mine.png';
	if (lower.includes('iron')) return '/assets/cards/iron_works.png';
	if (lower.includes('pottery')) return '/assets/cards/pottery.png';
	if (lower.includes('goods') || lower.includes('cotton')) return '/assets/cards/man_goods_or_cotton.png';
	return '/assets/grey-card.png';
}

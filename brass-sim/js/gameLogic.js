// ============================================================================
// Brass: Birmingham - Game Logic
// ============================================================================

(() => {
const __gameLogicData = (typeof module !== 'undefined' && module.exports)
    ? require('./gameData.js')
    : globalThis;

const {
    INDUSTRY_TYPES,
    ACTIONS,
    CARD_TYPES,
    INDUSTRY_DISPLAY,
    CITIES,
    MERCHANTS,
    CONNECTIONS,
    CANAL_LINK_COST,
    RAIL_LINK_COST,
    LOAN_AMOUNT,
    LOAN_INCOME_PENALTY,
    isSellableIndustry,
    isCity,
    isBreweryFarm,
} = __gameLogicData;

class GameLogic {
    constructor(gameState) {
        this.state = gameState;
    }

    // ========================================================================
    // Action Validation
    // ========================================================================

    canPerformAction(action, playerId) {
        const player = this.state.players[playerId];
        if (player.hand.length === 0) return false;

        switch (action) {
            case ACTIONS.BUILD: return this.getValidBuildTargets(playerId).length > 0;
            case ACTIONS.NETWORK: return this.getValidNetworkTargets(playerId).length > 0;
            case ACTIONS.DEVELOP: return this.canDevelop(playerId);
            case ACTIONS.SELL: return this.getValidSellTargets(playerId).length > 0;
            case ACTIONS.LOAN: return this.state.canTakeLoan(playerId);
            case ACTIONS.SCOUT: return this.canScout(playerId);
            case ACTIONS.PASS: return true; // Can always pass
            default: return false;
        }
    }

    // ========================================================================
    // BUILD Action
    // ========================================================================

    getValidBuildTargets(playerId) {
        const player = this.state.players[playerId];
        const targets = [];

        for (const [cityId, city] of Object.entries(CITIES)) {
            city.slots.forEach((slotTypes, slotIndex) => {
                const key = `${cityId}_${slotIndex}`;
                const existing = this.state.boardIndustries[key];

                // Check each industry type allowed in this slot
                const allowedTypes = Array.isArray(slotTypes) ? slotTypes : [slotTypes];

                for (const indType of allowedTypes) {
                    const nextTile = this.state.getNextTile(playerId, indType);
                    if (!nextTile) continue;

                    // Check era restrictions
                    if (this.state.era === ERA.CANAL && !nextTile.canalEra) continue;
                    if (this.state.era === ERA.RAIL && !nextTile.railEra) continue;

                    // Canal era: only one tile per location per player
                    if (this.state.era === ERA.CANAL) {
                        let hasOwnTile = false;
                        for (let i = 0; i < city.slots.length; i++) {
                            const k = `${cityId}_${i}`;
                            const t = this.state.boardIndustries[k];
                            if (t && t.playerId === playerId) {
                                hasOwnTile = true;
                                break;
                            }
                        }
                        if (hasOwnTile) continue;
                    }

                    // Check if slot is empty or can be overbuilt
                    if (existing) {
                        // Overbuilding rules
                        if (existing.playerId === playerId) {
                            // Own tile: can replace with same type, higher level
                            if (existing.type === indType && nextTile.level > existing.tileData.level) {
                                // OK - overbuilding own tile
                            } else {
                                continue;
                            }
                        } else {
                            // Opponent tile: can only replace Coal/Iron when 0 cubes of that resource exist
                            if (existing.type === INDUSTRY_TYPES.COAL_MINE ||
                                existing.type === INDUSTRY_TYPES.IRON_WORKS) {
                                // Check if there are 0 resource cubes of that type on entire board
                                if (!this.isResourceDepleted(existing.type)) continue;
                            } else {
                                continue; // Can't overbuild other opponent tiles
                            }
                        }
                    }

                    // Check if player can afford it
                    const cost = this.calculateBuildCost(playerId, indType, cityId);
                    if (cost === null) continue; // Can't afford resources

                    // Check card requirements
                    const hasValidCard = this.hasCardForBuild(playerId, cityId, indType);
                    if (!hasValidCard) continue;

                    targets.push({
                        cityId,
                        slotIndex,
                        industryType: indType,
                        tileData: nextTile,
                        cost,
                    });
                }
            });
        }

        return targets;
    }

    hasBuildNetwork(playerId,cityId){
        return this.state.isInNetwork(playerId,cityId)||(this.state.rulesVersion==='economy-v2'&&!Object.values(this.state.boardIndustries).some(t=>t.playerId===playerId)&&!Object.values(this.state.breweryFarmTiles).some(t=>t?.playerId===playerId)&&!Object.values(this.state.boardLinks).some(t=>t.playerId===playerId));
    }

    hasCardForBuild(playerId, cityId, industryType) {
        const player = this.state.players[playerId];
        const inNetwork = this.state.isInNetwork(playerId, cityId) || (this.state.rulesVersion==='economy-v2' && !Object.values(this.state.boardIndustries).some(t=>t.playerId===playerId) && !Object.values(this.state.breweryFarmTiles).some(t=>t?.playerId===playerId) && !Object.values(this.state.boardLinks).some(t=>t.playerId===playerId));
        for (const card of player.hand) {
            // Location card: can build at that location (no network needed)
            if (card.type === CARD_TYPES.LOCATION && card.location === cityId) return true;
            // Industry card: can build that industry at any location IN network
            if (card.type === CARD_TYPES.INDUSTRY && card.industryType === industryType && inNetwork) return true;
            // Wild location: acts as any location card
            if (card.type === CARD_TYPES.WILD_LOCATION) return true;
            // Wild industry: acts as any industry card (needs network)
            if (card.type === CARD_TYPES.WILD_INDUSTRY && inNetwork) return true;
        }
        return false;
    }

    calculateBuildCost(playerId, industryType, cityId) {
        const player = this.state.players[playerId];
        const tile = this.state.getNextTile(playerId, industryType);
        if (!tile) return null;

        let moneyCost = tile.cost;
        let coalNeeded = tile.costCoal;
        let ironNeeded = tile.costIron;

        // Find cheapest coal sources
        let coalCost = 0;
        if (coalNeeded > 0) {
            const sources = this.state.findCoalSource(cityId, playerId);
            let remaining = coalNeeded;
            for (const src of sources) {
                if (remaining <= 0) break;
                if (src.free) {
                    remaining--;
                } else {
                    coalCost += src.price;
                    remaining--;
                }
            }
            if (remaining > 0) return null; // Not enough coal
        }

        // Find cheapest iron sources
        let ironCost = 0;
        if (ironNeeded > 0) {
            const sources = this.state.findIronSource(playerId);
            let remaining = ironNeeded;
            for (const src of sources) {
                if (remaining <= 0) break;
                if (src.free) {
                    remaining--;
                } else {
                    ironCost += src.price;
                    remaining--;
                }
            }
            if (remaining > 0) return null; // Not enough iron
        }

        const totalCost = moneyCost + coalCost + ironCost;
        if (totalCost > player.money) return null;

        return {
            money: moneyCost,
            coal: coalNeeded,
            coalCost,
            iron: ironNeeded,
            ironCost,
            total: totalCost,
        };
    }

    executeBuild(playerId, cityId, slotIndex, industryType, cardIndex) {
        const player = this.state.players[playerId];
        const key = `${cityId}_${slotIndex}`;

        // Calculate and pay costs
        const cost = this.calculateBuildCost(playerId, industryType, cityId);
        if (!cost) return { success: false, message: 'Cannot afford this build' };

        // Get the tile to place after confirming the build is legal
        const tileData = this.state.useNextTile(playerId, industryType);
        if (!tileData) return { success: false, message: 'No tile available' };

        this.state.spendMoney(playerId, cost.total);

        // Consume coal
        if (cost.coal > 0) {
            let remaining = cost.coal;
            const sources = this.state.findCoalSource(cityId, playerId);
            for (const src of sources) {
                if (remaining <= 0) break;
                if (src.type === 'mine') {
                    this.state.consumeResource(src.key);
                    remaining--;
                } else if (src.type === 'market') {
                    this.state.coalMarket=Math.max(0,this.state.coalMarket-1);
                    remaining--;
                }
            }
        }

        // Consume iron
        if (cost.iron > 0) {
            let remaining = cost.iron;
            const sources = this.state.findIronSource(playerId);
            for (const src of sources) {
                if (remaining <= 0) break;
                if (src.type === 'works') {
                    this.state.consumeResource(src.key);
                    remaining--;
                } else if (src.type === 'market') {
                    this.state.ironMarket=Math.max(0,this.state.ironMarket-1);
                    remaining--;
                }
            }
        }

        // Remove the existing tile if overbuilding
        const existing = this.state.boardIndustries[key];
        if (existing) {
            // Overbuilt tile is removed from the game
        }

        // Place the tile
        this.state.boardIndustries[key] = {
            playerId,
            type: industryType,
            tileData: tileData,
            flipped: false,
            resourceCubes: tileData.resourceCubes || 0,
        };

        this.state.supplyNewIndustry(key);
        // Discard the used card
        this.discardCard(playerId, cardIndex);

        return { success: true, message: `Built ${INDUSTRY_DISPLAY[industryType].name} Level ${tileData.level} in ${CITIES[cityId].name}` };
    }

    isResourceDepleted(industryType) {
        // Check if there are 0 resource cubes of this type anywhere on the board
        if (industryType === INDUSTRY_TYPES.COAL_MINE) {
            if (this.state.coalMarket > 0) return false;
            for (const tile of Object.values(this.state.boardIndustries)) {
                if (tile.type === INDUSTRY_TYPES.COAL_MINE && !tile.flipped && tile.resourceCubes > 0) {
                    return false;
                }
            }
            return true;
        }
        if (industryType === INDUSTRY_TYPES.IRON_WORKS) {
            if (this.state.ironMarket > 0) return false;
            for (const tile of Object.values(this.state.boardIndustries)) {
                if (tile.type === INDUSTRY_TYPES.IRON_WORKS && !tile.flipped && tile.resourceCubes > 0) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    // ========================================================================
    // NETWORK Action
    // ========================================================================

    getValidNetworkTargets(playerId) {
        const player = this.state.players[playerId];
        const targets = [];
        const era = this.state.era;

        for (const conn of CONNECTIONS) {
            if (this.state.boardLinks[conn.id]) continue; // Already built

            // Check era
            if (era === ERA.CANAL && !conn.canal) continue;
            if (era === ERA.RAIL && !conn.rail) continue;

            // Check if player has link tiles remaining
            if (era === ERA.CANAL && player.linksRemaining.canal <= 0) continue;
            if (era === ERA.RAIL && player.linksRemaining.rail <= 0) continue;

            // Check network connection (at least one end must be in network)
            // Exception: first link of the game can go anywhere
            const hasAnyLinks = Object.values(this.state.boardLinks).some(l => l.playerId === playerId);
            const hasAnyIndustries = Object.values(this.state.boardIndustries).some(t => t.playerId === playerId);

            if (hasAnyLinks || hasAnyIndustries) {
                const end1InNetwork = this.state.isInNetwork(playerId, conn.cities[0]);
                const end2InNetwork = this.state.isInNetwork(playerId, conn.cities[1]);
                if (!end1InNetwork && !end2InNetwork) continue;
            }

            // Check cost
            let cost;
            if (era === ERA.CANAL) {
                cost = CANAL_LINK_COST;
            } else {
                cost = RAIL_LINK_COST;
                // Rail also needs coal
                const coalSources = this.state.findCoalSource(conn.cities[0], playerId);
                if (coalSources.length === 0) continue; // No coal available
                const coalSource = coalSources[0];
                cost += coalSource.free ? 0 : coalSource.price;
            }

            if (cost > player.money) continue;

            targets.push({
                connectionId: conn.id,
                cities: conn.cities,
                cost,
                type: era === ERA.CANAL ? 'canal' : 'rail',
            });
        }

        return targets;
    }

    executeNetwork(playerId, connectionId, cardIndex) {
        const player = this.state.players[playerId];
        const conn = CONNECTIONS.find(c => c.id === connectionId);
        if (!conn) return { success: false, message: 'Invalid connection' };

        const era = this.state.era;
        const linkType = era === ERA.CANAL ? 'canal' : 'rail';

        // Pay cost
        if (era === ERA.CANAL) {
            this.state.spendMoney(playerId, CANAL_LINK_COST);
        } else {
            let totalCost = RAIL_LINK_COST;
            // Consume coal for rail
            const coalSources = this.state.findCoalSource(conn.cities[0], playerId);
            if (coalSources.length > 0) {
                const src = coalSources[0];
                if (src.type === 'mine') {
                    this.state.consumeResource(src.key);
                } else {
                    // Buy coal from market - add market price to total
                    totalCost += src.price;
                    this.state.coalMarket=Math.max(0,this.state.coalMarket-1);
                }
            }
            this.state.spendMoney(playerId, totalCost);
        }

        // Place link
        this.state.boardLinks[connectionId] = {
            playerId,
            type: linkType,
        };

        // Reduce remaining links
        if (linkType === 'canal') {
            player.linksRemaining.canal--;
        } else {
            player.linksRemaining.rail--;
        }

        // Discard card
        this.discardCard(playerId, cardIndex);

        const city1 = CITIES[conn.cities[0]]?.name || MERCHANTS[conn.cities[0]]?.name || conn.cities[0];
        const city2 = CITIES[conn.cities[1]]?.name || MERCHANTS[conn.cities[1]]?.name || conn.cities[1];

        return { success: true, message: `Built ${linkType} link: ${city1} - ${city2}` };
    }

    // ========================================================================
    // DEVELOP Action
    // ========================================================================

    canDevelop(playerId) {
        const player = this.state.players[playerId];
        // Need iron to develop (1 iron per tile removed)
        const ironSources = this.state.findIronSource(playerId);
        if (ironSources.length === 0) return false;

        // Need at least one developable tile
        for (const [type, tiles] of Object.entries(player.industryTiles)) {
            const nextTile = tiles.find(t => !t.used);
            if (nextTile && nextTile.canDevelop) return true;
        }
        return false;
    }

    getDevelopableTypes(playerId) {
        const player = this.state.players[playerId];
        const types = [];
        for (const [type, tiles] of Object.entries(player.industryTiles)) {
            const nextTile = tiles.find(t => !t.used);
            if (nextTile && nextTile.canDevelop) {
                types.push({
                    type,
                    tile: nextTile,
                    name: INDUSTRY_DISPLAY[type].name,
                    level: nextTile.level,
                });
            }
        }
        return types;
    }

    executeDevelop(playerId, industryType1, industryType2, cardIndex) {
        // Develop removes 1 or 2 tiles from player mat (uses iron)
        // industryType2 can be null for single develop
        const player = this.state.players[playerId];

        if(this.state.rulesVersion==='economy-v2'){
            const used=new Map(),types=[industryType1,industryType2].filter(Boolean);
            for(const type of types){const offset=used.get(type)||0,tile=player.industryTiles[type]?.filter(t=>!t.used)[offset];if(!tile?.canDevelop)return {success:false,message:'Cannot develop this tile'};used.set(type,offset+1);}
            const sources=this.state.findIronSource(playerId).slice(0,types.length);
            if(sources.length<types.length||sources.reduce((sum,s)=>sum+(s.free?0:s.price),0)>player.money)return {success:false,message:'Cannot afford development'};
        }
        // Validate iron availability before proceeding
        const tilesToDevelop = industryType2 ? 2 : 1;
        const ironSources = this.state.findIronSource(playerId);
        if (ironSources.length < tilesToDevelop) {
            return { success: false, message: 'Not enough iron available' };
        }

        // Consume iron (1 per tile developed)
        for (let i = 0; i < tilesToDevelop; i++) {
            const src = ironSources[i];
            if (src.type === 'works') {
                this.state.consumeResource(src.key);
            } else {
                // Buy from market
                const price = this.state.getIronPrice();
                this.state.spendMoney(playerId, price);
                this.state.ironMarket=Math.max(0,this.state.ironMarket-1);
            }
        }

        // Remove tiles from player mat
        const tile1 = this.state.developTile(playerId, industryType1);
        let tile2 = null;
        if (industryType2) {
            tile2 = this.state.developTile(playerId, industryType2);
        }

        // Discard card
        this.discardCard(playerId, cardIndex);

        let msg = `Developed ${INDUSTRY_DISPLAY[industryType1].name}`;
        if (tile2) msg += ` and ${INDUSTRY_DISPLAY[industryType2].name}`;

        return { success: true, message: msg };
    }

    // ========================================================================
    // SELL Action
    // ========================================================================

    // Reserve actual cubes/merchant beer without applying a game transition.
    planSales(playerId, keys) {
        if(!keys.length||new Set(keys).size!==keys.length)return null;
        const reserved=new Map(),steps=[];
        for(const key of keys){
            const tile=this.state.boardIndustries[key];
            if(!tile||tile.playerId!==playerId||tile.flipped||!isSellableIndustry(tile.type))return null;
            const cityId=key.slice(0,key.lastIndexOf('_')),connected=this.state.getConnectedLocations(cityId);
            const merchants=this.state.merchantTiles.map((m,index)=>({m,index})).filter(({m})=>connected.has(m.location)&&(m.buys===null||m.buys===tile.type));
            let chosen=null;
            for(const {m,index} of merchants){
                const options=this.state.findBeerSources(cityId,playerId).filter(s=>s.type!=='merchant'||s.index===index);
                const beer=[];
                for(const src of options){
                    const id=src.type==='merchant'?`merchant:${src.index}`:src.key;
                    const source=src.type==='merchant'?null:src.key.startsWith('farm_')?this.state.breweryFarmTiles[src.key.slice(5)]:this.state.boardIndustries[src.key];
                    const available=(src.type==='merchant'?(m.hasBeer?1:0):source?.resourceCubes||0)-(reserved.get(id)||0);
                    for(let n=0;n<available&&beer.length<(tile.tileData.beersToSell||0);n++)beer.push({...src,id});
                }
                if(beer.length===(tile.tileData.beersToSell||0)){chosen={key,cityId,tile,merchant:index,beer};break;}
            }
            if(!chosen)return null;
            for(const src of chosen.beer)reserved.set(src.id,(reserved.get(src.id)||0)+1);
            steps.push(chosen);
        }
        return steps;
    }

    getValidSellBundles(playerId) {
        const targets=this.getValidSellTargets(playerId).sort((a,b)=>b.tile.tileData.vp-a.tile.tileData.vp),bundles=[],seen=new Set();
        const add=keys=>{const id=keys.slice().sort().join('|');if(keys.length<2||seen.has(id)||!this.planSales(playerId,keys))return;seen.add(id);bundles.push({keys,key:keys[0],tile:targets.find(t=>t.key===keys[0]).tile,cityId:targets.find(t=>t.key===keys[0]).cityId});};
        for(let i=0;i<targets.length;i++)for(let j=i+1;j<targets.length&&bundles.length<12;j++)add([targets[i].key,targets[j].key]);
        const all=[];for(const t of targets)if(this.planSales(playerId,all.concat(t.key)))all.push(t.key);add(all);
        return bundles;
    }

    getValidSellTargets(playerId) {
        if(this.state.rulesVersion==='economy-v2')return Object.keys(this.state.boardIndustries).flatMap(key=>{
            const plan=this.planSales(playerId,[key]);if(!plan)return [];
            const {cityId,tile}=plan[0];return [{key,cityId,tile,beerNeeded:tile.tileData.beersToSell||0}];
        });
        const targets = [];

        for (const [key, tile] of Object.entries(this.state.boardIndustries)) {
            if (tile.playerId !== playerId) continue;
            if (tile.flipped) continue;
            if (!isSellableIndustry(tile.type)) continue;

            // Check if player has beer available
            const beerNeeded = tile.tileData.beersToSell || 0;
            if (beerNeeded > 0) {
                const [cityId] = key.split('_');
                const beerSources = this.state.findBeerSources(cityId, playerId);
                if (beerSources.length < beerNeeded) continue;
            }

            // Check merchant connection for selling
            const [cityId] = key.split('_');
            const connected = this.state.getConnectedLocations(cityId);
            let hasMerchant = false;

            for (const mt of this.state.merchantTiles) {
                if (connected.has(mt.location)) {
                    if (mt.buys === null || mt.buys === tile.type) {
                        hasMerchant = true;
                        break;
                    }
                }
            }

            // Manufacturer III and VII have beersToSell = 0 and don't need merchants
            if (tile.tileData.beersToSell === 0) {
                hasMerchant = true; // These auto-sell
            }

            if (!hasMerchant) continue;

            targets.push({
                key,
                cityId,
                tile,
                beerNeeded,
            });
        }

        return targets;
    }

    executeSell(playerId, tileKeys, cardIndex) {
        // tileKeys: array of board keys to sell
        if(this.state.rulesVersion==='economy-v2'){
            const plan=this.planSales(playerId,tileKeys);if(!plan)return {success:false,message:'Cannot sell this combination with available beer and merchants'};
            for(const step of plan){
                for(const src of step.beer){
                    if(src.type!=='merchant'){this.state.consumeResource(src.key);continue;}
                    const mt=this.state.merchantTiles[src.index];mt.hasBeer=false;mt.bonusClaimed=true;
                    const bonus=MERCHANTS[mt.location],player=this.state.players[playerId];
                    if(bonus?.bonusType==='vp')player.vp+=bonus.bonusAmount;
                    if(bonus?.bonusType==='money')player.money+=bonus.bonusAmount;
                    if(bonus?.bonusType==='income')this.state.adjustIncome(playerId,bonus.bonusAmount);
                    if(bonus?.bonusType==='develop')this.applyFreeDevelop(playerId,bonus.bonusAmount);
                }
                step.tile.flipped=true;this.state.adjustIncome(playerId,step.tile.tileData.income);
            }
            this.discardCard(playerId,cardIndex);return {success:true,message:`Sold ${tileKeys.join(', ')}`};
        }
        const player = this.state.players[playerId];
        const results = [];

        for (const key of tileKeys) {
            const tile = this.state.boardIndustries[key];
            if (!tile || tile.playerId !== playerId || tile.flipped) continue;

            const beerNeeded = tile.tileData.beersToSell || 0;
            const [cityId] = key.split('_');

            // Consume beer
            if (beerNeeded > 0) {
                const beerSources = this.state.findBeerSources(cityId, playerId);
                for (let i = 0; i < beerNeeded && i < beerSources.length; i++) {
                    const src = beerSources[i];
                    if (src.type === 'merchant') {
                        this.state.merchantTiles[src.index].hasBeer = false;
                    } else {
                        this.state.consumeResource(src.key);
                    }
                }
            }

            // Flip tile
            tile.flipped = true;
            this.state.adjustIncome(playerId, tile.tileData.income);

            // Check for merchant bonus
            const connected = this.state.getConnectedLocations(cityId);
            for (const mt of this.state.merchantTiles) {
                if (!mt.bonusClaimed && connected.has(mt.location)) {
                    if (mt.buys === null || mt.buys === tile.type) {
                        mt.bonusClaimed = true;
                        // Apply bonus
                        const merchData = MERCHANTS[mt.location];
                        if (merchData) {
                            switch (merchData.bonusType) {
                                case 'vp':
                                    player.vp += merchData.bonusAmount;
                                    break;
                                case 'money':
                                    player.money += merchData.bonusAmount;
                                    break;
                                case 'income':
                                    this.state.adjustIncome(playerId, merchData.bonusAmount);
                                    break;
                                case 'develop':
                                    // Free develop: remove lowest tile from player mat (no iron cost)
                                    this.applyFreeDevelop(playerId, merchData.bonusAmount);
                                    break;
                            }
                        }
                        break; // Only one bonus per sell
                    }
                }
            }

            results.push(`Sold ${INDUSTRY_DISPLAY[tile.type].name} Lv${tile.tileData.level}`);
        }

        // Discard card
        this.discardCard(playerId, cardIndex);

        return { success: true, message: results.join(', ') };
    }

    // ========================================================================
    // LOAN Action
    // ========================================================================

    executeLoan(playerId, cardIndex) {
        const player = this.state.players[playerId];
        if(!this.state.takeLoan(playerId))return {success:false,message:'Cannot reduce income below -10'};

        this.discardCard(playerId, cardIndex);

        return { success: true, message: `Took £${LOAN_AMOUNT} loan (income -${LOAN_INCOME_PENALTY})` };
    }

    // ========================================================================
    // SCOUT Action
    // ========================================================================

    canScout(playerId) {
        const player = this.state.players[playerId];
        // Need at least 3 cards in hand (1 for action + 2 additional)
        if (player.hand.length < 3) return false;
        // Cannot have wild cards already
        if (player.hasWildLocation || player.hasWildIndustry) return false;
        // Must have wild cards available
        if (this.state.wildLocationPile <= 0 || this.state.wildIndustryPile <= 0) return false;
        return true;
    }

    executeScout(playerId, cardIndices) {
        // cardIndices: [actionCard, extraCard1, extraCard2] (3 cards total)
        const player = this.state.players[playerId];

        if (cardIndices.length !== 3) {
            return { success: false, message: 'Must discard exactly 3 cards' };
        }

        // Remove cards in reverse index order to maintain indices
        const sorted = [...cardIndices].sort((a, b) => b - a);
        for (const idx of sorted) {
            player.hand.splice(idx, 1);
        }

        // Give wild cards
        player.hand.push({
            type: CARD_TYPES.WILD_LOCATION,
            name: 'Wild Location',
        });
        player.hand.push({
            type: CARD_TYPES.WILD_INDUSTRY,
            name: 'Wild Industry',
        });

        player.hasWildLocation = true;
        player.hasWildIndustry = true;

        this.state.wildLocationPile--;
        this.state.wildIndustryPile--;

        return { success: true, message: 'Scouted: gained Wild Location + Wild Industry' };
    }

    // ========================================================================
    // PASS Action
    // ========================================================================

    executePass(playerId, cardIndex) {
        this.discardCard(playerId, cardIndex);
        return { success: true, message: 'Passed' };
    }

    // ========================================================================
    // Free Develop (merchant bonus)
    // ========================================================================

    applyFreeDevelop(playerId, count) {
        for (let i = 0; i < count; i++) {
            // Find the lowest-level developable tile on the player's mat
            const types = this.getDevelopableTypes(playerId);
            if (types.length === 0) break;
            // Pick the lowest level tile
            types.sort((a, b) => a.level - b.level);
            this.state.developTile(playerId, types[0].type);
        }
    }

    // ========================================================================
    // Disabled Reason for Action Buttons
    // ========================================================================

    getDisabledReason(action, playerId) {
        const player = this.state.players[playerId];
        if (player.hand.length === 0) return 'Hand is empty';

        switch (action) {
            case ACTIONS.BUILD:
                if (this.getValidBuildTargets(playerId).length === 0) {
                    return 'No valid build locations';
                }
                return null;
            case ACTIONS.NETWORK: {
                const era = this.state.era;
                if (era === ERA.CANAL && player.linksRemaining.canal <= 0) return 'No canal links remaining';
                if (era === ERA.RAIL && player.linksRemaining.rail <= 0) return 'No rail links remaining';
                if (this.getValidNetworkTargets(playerId).length === 0) return 'No valid connections available';
                return null;
            }
            case ACTIONS.DEVELOP: {
                const ironSources = this.state.findIronSource(playerId);
                if (ironSources.length === 0) return 'No iron available';
                if (!this.canDevelop(playerId)) return 'No developable tiles on your mat';
                return null;
            }
            case ACTIONS.SELL:
                if (this.getValidSellTargets(playerId).length === 0) return 'No industries ready to sell';
                return null;
            case ACTIONS.LOAN:
                return null;
            case ACTIONS.SCOUT:
                if (player.hand.length < 3) return 'Need at least 3 cards';
                if (player.hasWildLocation || player.hasWildIndustry) return 'Already have wild cards';
                if (this.state.wildLocationPile <= 0 || this.state.wildIndustryPile <= 0) return 'No wild cards available';
                return null;
            case ACTIONS.PASS:
                return null;
            default:
                return 'Unknown action';
        }
    }

    // ========================================================================
    // Card Management
    // ========================================================================

    discardCard(playerId, cardIndex) {
        const player = this.state.players[playerId];
        if (cardIndex < 0 || cardIndex >= player.hand.length) return;

        const card = player.hand[cardIndex];

        // Wild cards go back to their piles
        if (card.type === CARD_TYPES.WILD_LOCATION) {
            this.state.wildLocationPile++;
            player.hasWildLocation = false;
        } else if (card.type === CARD_TYPES.WILD_INDUSTRY) {
            this.state.wildIndustryPile++;
            player.hasWildIndustry = false;
        }

        player.hand.splice(cardIndex, 1);
    }

    // ========================================================================
    // Get valid cards for an action
    // ========================================================================

    getValidCardsForAction(playerId, action, target = null) {
        const player = this.state.players[playerId];
        const validIndices = [];

        player.hand.forEach((card, idx) => {
            switch (action) {
                case ACTIONS.BUILD:
                    if (target) {
                        // Check if card matches the build target
                        if (card.type === CARD_TYPES.LOCATION && card.location === target.cityId) {
                            validIndices.push(idx);
                        } else if (card.type === CARD_TYPES.INDUSTRY && card.industryType === target.industryType) {
                            // Industry card: target must be in network
                            if (this.hasBuildNetwork(playerId, target.cityId)) {
                                validIndices.push(idx);
                            }
                        } else if (card.type === CARD_TYPES.WILD_LOCATION) {
                            validIndices.push(idx);
                        } else if (card.type === CARD_TYPES.WILD_INDUSTRY) {
                            if (this.hasBuildNetwork(playerId, target.cityId)) {
                                validIndices.push(idx);
                            }
                        }
                    } else {
                        // Any card can potentially be used for build
                        validIndices.push(idx);
                    }
                    break;

                case ACTIONS.NETWORK:
                case ACTIONS.DEVELOP:
                case ACTIONS.SELL:
                case ACTIONS.LOAN:
                case ACTIONS.PASS:
                    // Any card can be discarded for these actions
                    validIndices.push(idx);
                    break;

                case ACTIONS.SCOUT:
                    // All cards are candidates for scout discard
                    validIndices.push(idx);
                    break;
            }
        });

        return validIndices;
    }
}

if (typeof globalThis !== 'undefined') {
    globalThis.GameLogic = GameLogic;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = GameLogic;
}
})();

(function(root){
    'use strict';
    const E=typeof module!=='undefined'&&module.exports?require('./encoding'):root.BrassEncoding;
    const D=typeof module!=='undefined'&&module.exports?require('../js/gameData'):root;
    const topology=E.schema.links.map(id=>{
        const cities=D.CONNECTIONS.find(c=>c.id===id).cities;
        return {slots:E.schema.slots.flatMap((s,i)=>cities.includes(s.startsWith('farm:')?s.slice(5):s.slice(0,s.lastIndexOf('_')))?[i]:[]),merchantVP:cities.filter(D.isMerchantLocation).length*2};
    });
    const playerKeys=['money','income','vp','handSize','spent','canal','rail','wildLocation','wildIndustry'];
    const globalKeys=['era','round','players','deckSize','actionsThisTurn','actionsPerTurn','gameOver','coalMarket','ironMarket'];
    const clamp=x=>Math.max(0,Math.min(1,x));
    function features(v,actor,version='brass-value-v1'){
        if(!['brass-value-v1','brass-value-v2'].includes(version))throw Error('Unknown value feature version');
        const get=name=>v[E.indices[name]],raw=name=>E.raw(v,name),types=E.schema.types;
        const slots=E.schema.slots.map(s=>({owner:Math.round(raw(`slot.${s}.owner`)),type:Math.round(raw(`slot.${s}.type`)),
            keep:raw(`slot.${s}.level`)>=1.5,flip:clamp(raw(`slot.${s}.flipped`)),vp:Math.max(0,raw(`slot.${s}.vp`)),cubes:Math.max(0,raw(`slot.${s}.cubes`))}));
        const links=E.schema.links.map(s=>({owner:Math.round(raw(`link.${s}.owner`)),rail:clamp(raw(`link.${s}.rail`))}));
        const roles=Array.from({length:4},(_,p)=>{
            const owned=slots.filter(s=>s.owner===p+1),value=playerKeys.map(k=>get(`p${p}.${k}`)).concat(types.map(t=>get(`p${p}.used.${t}`)));
            for(let t=1;t<=types.length;t++)for(const keep of [false,true])for(const sold of [false,true])value.push(owned.filter(s=>s.type===t&&s.keep===keep).reduce((n,s)=>n+(sold?s.flip:1-s.flip),0)/4);
            value.push(owned.reduce((n,s)=>n+s.vp*s.flip,0)/100,owned.reduce((n,s)=>n+s.vp*(1-s.flip),0)/100);
            for(let t=1;t<=types.length;t++)value.push(owned.filter(s=>s.type===t).reduce((n,s)=>n+s.cubes,0)/10);
            value.push(links.filter(l=>l.owner===p+1).reduce((n,l)=>n+1-l.rail,0)/14,links.filter(l=>l.owner===p+1).reduce((n,l)=>n+l.rail,0)/14);
            return value;
        });
        const count=Math.max(1,Math.round(raw('players'))-1),opponents=roles[actor].map((_,i)=>roles.reduce((n,row,p)=>n+(p===actor?0:row[i]),0)/count);
        const base=globalKeys.map(get).concat(roles[actor],opponents,[get(`current.${actor}`),Math.max(...[0,1,2,3].filter(p=>p!==actor).map(p=>get(`p${p}.vp`)))]);
        if(version==='brass-value-v1')return base;
        // Observable geography makes the value of a connection distinguishable
        // from its count. Future flips remain uncertain and are learned from play.
        const icon=E.schema.slots.map(s=>Math.max(0,raw(`slot.${s}.linkVP`)));
        const extra=roles.map((_,p)=>{
            let scored=0,potential=0,ownScored=0,ownPotential=0;
            links.forEach((l,k)=>{if(l.owner!==p+1)return;scored+=topology[k].merchantVP;
                for(const i of topology[k].slots){const s=slots[i];if(s.owner<1||s.owner>4)continue;
                    const a=icon[i]*s.flip,b=icon[i]*(1-s.flip);scored+=a;potential+=b;
                    if(s.owner===p+1){ownScored+=a;ownPotential+=b;}
                }
            });
            const owned=slots.filter(s=>s.owner===p+1),sum=fn=>owned.reduce((n,s)=>n+fn(s),0);
            const sellable=s=>['cottonMill','manufacturer','pottery'].includes(E.schema.types[s.type-1]);
            return [scored,potential,sum(s=>s.keep?s.vp*s.flip:0),sum(s=>s.keep?s.vp*(1-s.flip):0),
                ownScored,ownPotential,sum(s=>sellable(s)?s.vp*(1-s.flip):0),sum(s=>sellable(s)&&s.keep?s.vp*(1-s.flip):0)].map(x=>x/100);
        });
        return base.concat(extra[actor],extra[actor].map((_,i)=>extra.reduce((n,row,p)=>n+(p===actor?0:row[i]),0)/count));
    }
    function scoreAnchor(v,actor){
        let total=E.raw(v,`p${actor}.vp`);
        if(E.raw(v,'gameOver')>=.5)return total;
        const owners=E.schema.slots.map(s=>Math.round(E.raw(v,`slot.${s}.owner`)));
        const flips=E.schema.slots.map(s=>clamp(E.raw(v,`slot.${s}.flipped`)));
        E.schema.slots.forEach((s,i)=>{if(owners[i]===actor+1)total+=Math.max(0,E.raw(v,`slot.${s}.vp`))*flips[i];});
        E.schema.links.forEach((id,k)=>{if(Math.round(E.raw(v,`link.${id}.owner`))!==actor+1)return;
            total+=topology[k].merchantVP;for(const i of topology[k].slots)if(owners[i]>=1&&owners[i]<=4)total+=Math.max(0,E.raw(v,`slot.${E.schema.slots[i]}.linkVP`))*flips[i];
        });
        return total;
    }
    if(typeof module!=='undefined'&&module.exports)module.exports={features,topology,scoreAnchor};else root.BrassValueFeatures={features,topology,scoreAnchor};
})(globalThis);

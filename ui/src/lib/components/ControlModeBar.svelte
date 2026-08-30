<script lang="ts">
	import { Bot, Eye, Hand, UserRound } from 'lucide-svelte';
	import type { GameControlMode, Player } from '$lib/types';
	import { createEventDispatcher } from 'svelte';

	export let mode: GameControlMode;
	export let humanPlayerIndex: number;
	export let players: Player[] = [];
	export let currentPlayerIndex: number;
	export let disabled = false;

	const dispatch = createEventDispatcher<{
		change: { mode: GameControlMode; humanPlayerIndex: number };
	}>();

	$: currentPlayer = players.find(player => player.index === currentPlayerIndex) ?? null;
	$: humanPlayer = players.find(player => player.index === humanPlayerIndex) ?? players[0] ?? null;
	$: isAiTurn = mode === 'ai-vs-ai'
		|| (mode === 'human-vs-ai' && currentPlayerIndex !== humanPlayerIndex);
	$: statusLabel = mode === 'manual' ? '手动操作' : isAiTurn ? 'AI 操作' : '你操作';

	function changeMode(event: Event) {
		mode = (event.currentTarget as HTMLSelectElement).value as GameControlMode;
		dispatch('change', { mode, humanPlayerIndex });
	}

	function changeHumanPlayer(event: Event) {
		humanPlayerIndex = Number((event.currentTarget as HTMLSelectElement).value);
		dispatch('change', { mode, humanPlayerIndex });
	}
</script>

<section
	class="control-mode-bar"
	class:ai-turn={isAiTurn}
	class:manual={mode === 'manual'}
	style="--player-color: {currentPlayer?.color ?? '#6a706c'}"
	aria-label="对局控制权"
>
	<div class="control-status">
		<span class="status-icon" aria-hidden="true">
			{#if mode === 'manual'}
				<Hand size={16} />
			{:else if isAiTurn}
				<Bot size={16} />
			{:else}
				<UserRound size={16} />
			{/if}
		</span>
		<span class="status-copy">
			<strong>{statusLabel}</strong>
			<span>{currentPlayer?.name ?? '等待对局'}</span>
		</span>
	</div>

	<div class="control-selects">
		<label>
			<span>模式</span>
			<select value={mode} on:change={changeMode} {disabled} aria-label="对局模式">
				<option value="human-vs-ai">人机对练</option>
				<option value="ai-vs-ai" disabled={players.length !== 2}>AI 观战</option>
				<option value="manual">全部手动</option>
			</select>
		</label>
		{#if mode === 'human-vs-ai'}
			<label>
				<span>你的席位</span>
				<select value={humanPlayerIndex} on:change={changeHumanPlayer} {disabled} aria-label="你的席位">
					{#each players as player}
						<option value={player.index}>{player.name}</option>
					{/each}
				</select>
			</label>
		{:else}
			<div class="mode-mark" title={mode === 'ai-vs-ai' ? 'AI 观战' : '全部手动'}>
				{#if mode === 'ai-vs-ai'}<Eye size={15} aria-hidden="true" />{:else}<Hand size={15} aria-hidden="true" />{/if}
			</div>
		{/if}
	</div>
</section>

<style>
	.control-mode-bar {
		min-height: 60px;
		display: grid;
		grid-template-columns: minmax(94px, 0.8fr) minmax(190px, 1.4fr);
		align-items: center;
		gap: 10px;
		padding: 7px 12px;
		border-bottom: 1px solid #cdd2ce;
		box-shadow: inset 3px 0 0 var(--player-color);
		background: #f5f6f4;
		color: #202321;
	}
	.control-mode-bar.ai-turn { background: #f7f2ef; }
	.control-mode-bar.manual { background: #f1f3f1; }
	.control-status { min-width: 0; display: flex; align-items: center; gap: 7px; }
	.status-icon {
		width: 28px;
		height: 28px;
		flex: 0 0 auto;
		display: grid;
		place-items: center;
		border: 1px solid #c4c9c5;
		border-radius: 50%;
		background: #fff;
		color: #087f5b;
	}
	.ai-turn .status-icon { color: #b54031; border-color: #d8b8b1; }
	.manual .status-icon { color: #5f6661; }
	.status-copy { min-width: 0; display: flex; flex-direction: column; }
	.status-copy strong { font-size: 11px; line-height: 1.2; }
	.status-copy span { margin-top: 2px; overflow: hidden; color: #727873; font-size: 10px; text-overflow: ellipsis; white-space: nowrap; }
	.control-selects { min-width: 0; display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 6px; align-items: end; }
	.control-selects label { min-width: 0; display: flex; flex-direction: column; gap: 2px; color: #727873; font-size: 9px; font-weight: 700; }
	.control-selects select {
		width: 100%;
		min-width: 0;
		height: 29px;
		border: 1px solid #c5cac6;
		border-radius: 5px;
		background: #fff;
		color: #303532;
		padding: 0 22px 0 7px;
		font-size: 10px;
		font-weight: 650;
	}
	.control-selects select:focus { outline: 2px solid rgba(8, 127, 91, .18); border-color: #087f5b; }
	.control-selects select:disabled { opacity: .55; cursor: not-allowed; }
	.mode-mark {
		height: 29px;
		display: grid;
		place-items: center;
		align-self: end;
		border: 1px solid #d0d4d1;
		border-radius: 5px;
		background: #fff;
		color: #6a706c;
	}
	@media (max-width: 400px) {
		.control-mode-bar { grid-template-columns: 92px minmax(0, 1fr); padding-inline: 10px; }
	}
</style>

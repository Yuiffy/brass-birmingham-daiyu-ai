use serde::Serialize;

use crate::board::resources::{BeerSellSource, BreweryBeerSource, ResourceSource};
use crate::core::types::{ActionType, Card, Era};
use crate::game::framework::{ActionIntent, NetworkMode};
use crate::game::rule_ai::RuleScoreBreakdown;
use crate::game::runner::GameRunner;
use crate::game::search::{
    RootActionEstimate, RootSearchReport, BATCHED_NEURAL_PUCT_METHOD, NEURAL_TREE_VALUE_SOURCE,
    ROOT_PUCT_ACTION_VALUE_METHOD, ROOT_PUCT_METHOD, RULE_DECISION_TREE_METHOD,
    RULE_ECONOMIC_METHOD, RULE_IMMEDIATE_SCORE_SOURCE, SUCCESSOR_MODEL_VALUE_SOURCE,
};

use super::serialize::{format_card_label, industry_str, town_name_for_bl};

#[derive(Debug, Clone, Serialize)]
pub struct AnalysisJson {
    pub revision: u64,
    pub method: String,
    pub method_label: &'static str,
    pub value_source: String,
    pub model_id: Option<String>,
    pub root_model_shared_win_rate: Option<f64>,
    pub root_model_victory_point_margin: Option<f64>,
    pub root_player: usize,
    pub requested_simulations: u64,
    pub completed_simulations: u64,
    pub root_action_count: usize,
    pub evaluated_action_count: usize,
    pub visited_action_count: usize,
    pub all_root_actions_evaluated: bool,
    pub max_search_depth: Option<usize>,
    pub neural_leaf_evaluations: Option<u64>,
    pub inference_batches: Option<u64>,
    pub coverage: f64,
    pub elapsed_ms: u128,
    pub recommendations: Vec<AnalysisCandidateJson>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnalysisCandidateJson {
    pub rank: usize,
    pub action_key: String,
    pub action_type: &'static str,
    pub action_label: &'static str,
    pub summary: String,
    pub selections: Vec<String>,
    pub visits: u64,
    pub visit_share: f64,
    pub value_source: String,
    pub value_sample_count: u64,
    pub estimated_shared_win_rate: f64,
    pub estimated_outright_win_rate: Option<f64>,
    pub estimated_tied_first_rate: Option<f64>,
    pub average_final_victory_points: Option<f64>,
    pub average_victory_point_margin: f64,
    pub shared_win_rate_standard_error: Option<f64>,
    pub policy_probability: Option<f64>,
    pub calibrated_win_rate: Option<f64>,
    pub rule_score: Option<f64>,
    pub rule_score_breakdown: Option<RuleScoreBreakdown>,
    pub immediate_effect: ImmediateEffectJson,
    pub highlights: AnalysisHighlightsJson,
    pub sample_random_continuation: SampleContinuationJson,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImmediateEffectJson {
    pub money_delta: i32,
    pub income_level_delta: i16,
    pub victory_points_delta: i32,
    pub visible_victory_points_delta: i32,
    pub potential_era_victory_points_delta: i32,
    pub hand_size_delta: i16,
    pub round_spend_delta: i32,
    pub roads_on_board_delta: i32,
    pub buildings_on_board_delta: i32,
    pub flipped_buildings_delta: i32,
    pub draw_deck_size_delta: i32,
    pub market_coal_delta: i16,
    pub market_iron_delta: i16,
    pub wild_location_pool_delta: i16,
    pub wild_industry_pool_delta: i16,
    pub turn_advanced: bool,
    pub phase_changed: bool,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct AnalysisHighlightsJson {
    pub building_locations: Vec<usize>,
    pub source_locations: Vec<usize>,
    pub roads: Vec<usize>,
    pub merchant_slots: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SampleContinuationJson {
    pub label: &'static str,
    pub steps: Vec<SampleContinuationStepJson>,
    pub final_victory_points: Vec<u16>,
    pub official_winners: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SampleContinuationStepJson {
    pub action_number: u32,
    pub player_idx: usize,
    pub action_type: &'static str,
    pub action_label: &'static str,
    pub action_key: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExplanationJson {
    pub revision: u64,
    pub action_key: String,
    pub question: String,
    pub answer: String,
    pub evidence: Vec<String>,
    pub caveat: String,
}

pub fn serialize_analysis(
    report: &RootSearchReport,
    runner: &GameRunner,
    revision: u64,
    elapsed_ms: u128,
) -> AnalysisJson {
    serialize_analysis_for_observer(report, runner, revision, elapsed_ms, true)
}

pub fn serialize_analysis_for_observer(
    report: &RootSearchReport,
    runner: &GameRunner,
    revision: u64,
    elapsed_ms: u128,
    reveal_private_cards: bool,
) -> AnalysisJson {
    let root_action_count = report.root_action_count;
    AnalysisJson {
        revision,
        method: report.method.clone(),
        method_label: if report.method == crate::game::trained_ai::METHOD {
            "学习增强 v2 · 真人策略与手牌评估"
        } else if report.method == BATCHED_NEURAL_PUCT_METHOD {
            "策略价值网络 · 多层 PUCT · 隐藏牌确定化"
        } else if report.method == RULE_ECONOMIC_METHOD {
            "经济规划 v2 · 回合组合与资金周转"
        } else if report.method == RULE_DECISION_TREE_METHOD {
            "CPU 规则决策树 · 同回合浅层前瞻（无终局续弈）"
        } else if report.method == ROOT_PUCT_ACTION_VALUE_METHOD {
            "策略网络先验 · 根节点 PUCT · 批量后继价值"
        } else if report.method == ROOT_PUCT_METHOD {
            "策略网络先验 · 根节点 PUCT · 隐藏牌确定化续弈"
        } else {
            "隐藏牌确定化 · 根节点 UCB · 随机续弈"
        },
        value_source: report.value_source.clone(),
        model_id: report.model_id.clone(),
        root_model_shared_win_rate: report.root_model_shared_win_rate,
        root_model_victory_point_margin: report.root_model_victory_point_margin,
        root_player: report.root_player,
        requested_simulations: report.requested_simulations,
        completed_simulations: report.completed_simulations,
        root_action_count,
        evaluated_action_count: report.evaluated_action_count,
        visited_action_count: report.visited_action_count,
        all_root_actions_evaluated: report.all_root_actions_evaluated,
        max_search_depth: report.max_search_depth,
        neural_leaf_evaluations: report.neural_leaf_evaluations,
        inference_batches: report.inference_batches,
        coverage: if root_action_count == 0 {
            0.0
        } else {
            report.evaluated_action_count as f64 / root_action_count as f64
        },
        elapsed_ms,
        recommendations: report
            .recommendations
            .iter()
            .map(|candidate| {
                serialize_candidate(candidate, runner, report.root_player, reveal_private_cards)
            })
            .collect(),
    }
}

pub fn explain_analysis_question(
    report: &RootSearchReport,
    runner: &GameRunner,
    revision: u64,
    action_key: &str,
    question: &str,
) -> Result<ExplanationJson, String> {
    explain_analysis_question_for_observer(report, runner, revision, action_key, question, true)
}

pub fn explain_analysis_question_for_observer(
    report: &RootSearchReport,
    runner: &GameRunner,
    revision: u64,
    action_key: &str,
    question: &str,
    reveal_private_cards: bool,
) -> Result<ExplanationJson, String> {
    let candidate = report
        .recommendations
        .iter()
        .find(|candidate| candidate.action_key == action_key)
        .ok_or_else(|| "the selected recommendation is not in the cached analysis".to_string())?;
    let top = report.recommendations.first();
    let serialized =
        serialize_candidate(candidate, runner, report.root_player, reveal_private_cards);
    if candidate.value_source == crate::game::trained_ai::VALUE_SOURCE {
        let effect = effect_sentence(&serialized.immediate_effect);
        let caveat = "评分结合冻结教师价值网络与真人策略评估，不是胜率或保证的终局分数；主站规则与实验室不同，均分需单独测量。";
        let mut answer = format!("这步排在第 {}，两步前瞻后的局面评分为 {:.2}。{} 评分由 20% 教师网络和 80% 真人策略组成，考虑产业翻面、线路收益、资金周转、剩余动作与手牌可达性。", candidate.rank, candidate.rule_score.unwrap_or(0.0), effect);
        if let Some(other) = report
            .recommendations
            .iter()
            .find(|c| c.action_key != action_key)
        {
            answer.push_str(&format!(
                "另一候选的局面评分为 {:.2}。",
                other.rule_score.unwrap_or(0.0)
            ));
        }
        return Ok(ExplanationJson {
            revision, action_key: action_key.into(), question: question.into(), answer,
            evidence: vec![format!("动作：{}", serialized.summary), format!("模型：{}", report.model_id.as_deref().unwrap_or("未知")),
                format!("候选覆盖：{}/{}；最大深度：{}", report.evaluated_action_count, report.root_action_count, report.max_search_depth.unwrap_or(0)),
                "按原生合法动作推进；未知手牌与未来牌堆采用确定化采样。前瞻可能受候选裁剪、隐藏牌样本和模型迁移误差影响。".into()],
            caveat: caveat.into(),
        });
    }
    let question_lower = question.to_lowercase();
    let asks_probability = question_lower.contains("胜率")
        || question_lower.contains("概率")
        || question_lower.contains("percent")
        || question_lower.contains("rate");
    let asks_risk = question_lower.contains("风险")
        || question_lower.contains("缺点")
        || question_lower.contains("坏")
        || question_lower.contains("不稳");
    let asks_resources = question_lower.contains("资源")
        || question_lower.contains("钱")
        || question_lower.contains("收入")
        || question_lower.contains("煤")
        || question_lower.contains("铁")
        || question_lower.contains("啤酒");
    let mentioned_choice_ranks = choice_ranks_mentioned(&question_lower);
    let asks_comparison = !mentioned_choice_ranks.is_empty()
        || question_lower.contains("相比")
        || question_lower.contains("比较")
        || question_lower.contains("对比")
        || question_lower.contains("为什么不是")
        || question_lower.contains("另一个");

    let visits = candidate.visits;
    let win_rate = format_percent(candidate.estimated_shared_win_rate);
    let visit_share = format_percent(candidate.visit_share);
    let standard_error = candidate
        .shared_win_rate_standard_error
        .map(format_percent)
        .unwrap_or_else(|| "样本不足".to_string());
    let effect = effect_sentence(&serialized.immediate_effect);
    let policy_prior = candidate.policy_probability.map(format_percent);
    let uses_successor_model = candidate.value_source == SUCCESSOR_MODEL_VALUE_SOURCE;
    let uses_neural_tree = candidate.value_source == NEURAL_TREE_VALUE_SOURCE;
    let uses_rule = candidate.value_source == RULE_IMMEDIATE_SCORE_SOURCE;
    let mut paragraphs = vec![if uses_neural_tree {
        format!(
            "这步排在第 {}：根策略先验为 {}，多层 PUCT 给了它 {} 次访问（占全部模拟的 {}）；{} 次树内价值回传后的共享胜分估计为 {}。{}",
            candidate.rank,
            policy_prior.as_deref().unwrap_or("不可用"),
            visits,
            visit_share,
            candidate.value_sample_count,
            win_rate,
            effect
        )
    } else if uses_successor_model {
        format!(
            "这步排在第 {}：策略网络先验为 {}，PUCT 搜索给了它 {} 次访问（占全部模拟的 {}）；{} 个隐藏牌确定化后继局面的模型共享胜分估计为 {}。{}",
            candidate.rank,
            policy_prior.as_deref().unwrap_or("不可用"),
            visits,
            visit_share,
            candidate.value_sample_count,
            win_rate,
            effect
        )
    } else if report.method == RULE_ECONOMIC_METHOD {
        format!(
            "这步排在第 {}：根据已兑现分数、跨时代产业收益、出售所需的路线和啤酒、现金储备及弃牌机会成本，经济评分为 {:+.2}。{}",
            candidate.rank,
            candidate.rule_score.unwrap_or(candidate.average_victory_point_margin),
            effect
        )
    } else if uses_rule {
        format!(
            "这步排在第 {}：CPU 规则决策树先评估所有合法动作，再对少量候选检查同回合下一动作；综合 VP、收入、现金、产业、真实路线、资源和弃牌机会成本后的规则评分为 {:+.2}。{}",
            candidate.rank,
            candidate
                .rule_score
                .unwrap_or(candidate.average_victory_point_margin),
            effect
        )
    } else {
        match &policy_prior {
            Some(prior) => format!(
                "这步排在第 {}：策略网络先验为 {}，PUCT 搜索给了它 {} 次访问（占全部模拟的 {}），随机续弈样本共享胜分率为 {}。{}",
                candidate.rank, prior, visits, visit_share, win_rate, effect
            ),
            None => format!(
                "这步排在第 {}，因为根节点搜索给了它 {} 次访问（占全部模拟的 {}），样本共享胜分率为 {}。{}",
                candidate.rank, visits, visit_share, win_rate, effect
            ),
        }
    }];

    if asks_probability {
        paragraphs.push(if uses_rule {
            format!(
                "这里的 {:+.2} 是当前局面的相对规则评分，不是胜率。评分只比较本次合法动作的前后状态，没有滚到结局，也没有使用神经网络或 GPU；真实胜率仍需独立对局校准。",
                candidate
                    .rule_score
                    .unwrap_or(candidate.average_victory_point_margin)
            )
        } else if uses_neural_tree {
            format!(
                "这里的 {} 是多层 PUCT 沿搜索树回传到根节点的共享胜分均值，当前实际最大深度为 {}，共完成 {} 个神经叶估值；多人局会把非根玩家的模型胜分按其余竞争者平均分配后换回根视角。它尚未经过实战校准，不能当作精确胜率。",
                win_rate,
                report.max_search_depth.unwrap_or(0),
                report.neural_leaf_evaluations.unwrap_or(0),
            )
        } else if uses_successor_model {
            format!(
                "这里的 {} 是后继局面价值网络均值，标准误约为 {}；轮到对手时已按二人零和关系换回当前玩家视角。它尚未经过实战校准，也还不是深层树搜索的终局胜率。",
                win_rate, standard_error
            )
        } else { match &policy_prior {
            Some(prior) => format!(
                "策略先验 {} 表示模型在搜索前对这步分配的相对概率；{} 是随机续弈样本均值，标准误约为 {}。两者都没有经过实战校准，不能当作真实对局的精确胜率。",
                prior, win_rate, standard_error
            ),
            None => format!(
                "这里的 {} 是随机续弈样本均值，标准误约为 {}；策略网络概率和校准胜率目前都没有，因此不能把它理解为真实对局中的精确胜率。",
                win_rate, standard_error
            ),
        }});
    }
    if asks_resources {
        paragraphs.push(format!("从立即变化看，{}", effect));
    }
    if asks_risk {
        paragraphs.push(risk_sentence(candidate, report));
    }
    if asks_comparison || candidate.rank > 1 {
        let comparison = mentioned_choice_ranks
            .iter()
            .copied()
            .find(|rank| *rank != candidate.rank)
            .and_then(|rank| {
                report
                    .recommendations
                    .iter()
                    .find(|other| other.rank == rank)
            })
            .or_else(|| {
                if candidate.rank == 1 {
                    report
                        .recommendations
                        .iter()
                        .find(|other| other.action_key != candidate.action_key)
                } else {
                    top.filter(|top| top.action_key != candidate.action_key)
                }
            });
        if let Some(other) = comparison {
            let candidate_label = choice_rank_label(candidate.rank);
            let other_label = choice_rank_label(other.rank);
            let other_serialized =
                serialize_candidate(other, runner, report.root_player, reveal_private_cards);
            if uses_rule {
                paragraphs.push(format!(
                    "与{}“{}”相比，{}的规则评分为 {:+.2} 对 {:+.2}。排序直接来自同一组阶段权重和安全约束，不包含终局续弈或神经网络估值。",
                    other_label,
                    describe_action(
                        &other.action.intent,
                        runner,
                        report.root_player,
                        reveal_private_cards,
                    )
                    .0,
                    candidate_label,
                    candidate.rule_score.unwrap_or(candidate.average_victory_point_margin),
                    other.rule_score.unwrap_or(other.average_victory_point_margin),
                ));
            } else {
                paragraphs.push(format!(
                    "与{}“{}”相比，{}的搜索访问数为 {} 对 {}，{}为 {} 对 {}。当前排序先按搜索访问数，再用价值估计打破访问数平局；差距仍受模型误差和隐藏牌样本量影响。",
                    other_label,
                    describe_action(
                        &other.action.intent,
                        runner,
                        report.root_player,
                        reveal_private_cards,
                    )
                    .0,
                    candidate_label,
                    candidate.visits,
                    other.visits,
                    if uses_neural_tree {
                        "多层树回传共享胜分估计"
                    } else if uses_successor_model {
                        "后继模型共享胜分估计"
                    } else {
                        "随机续弈共享胜分率"
                    },
                    win_rate,
                    format_percent(other.estimated_shared_win_rate),
                ));
                if let (Some(candidate_prior), Some(other_prior)) =
                    (candidate.policy_probability, other.policy_probability)
                {
                    paragraphs.push(format!(
                        "模型搜索前给{}和{}的策略先验分别为 {}、{}；访问差异同时反映先验和{}。",
                        candidate_label,
                        other_label,
                        format_percent(candidate_prior),
                        format_percent(other_prior),
                        if uses_neural_tree {
                            "多层树中的对手应对与叶节点价值反馈"
                        } else if uses_successor_model {
                            "后继价值反馈"
                        } else {
                            "续弈反馈"
                        }
                    ));
                }
            }
            paragraphs.push(format!(
                "{}的即时结果：{}{}的即时结果：{}",
                candidate_label,
                effect,
                other_label,
                effect_sentence(&other_serialized.immediate_effect)
            ));
        }
    }
    if paragraphs.len() == 1 {
        if uses_rule {
            paragraphs.push(
                "这是 CPU 浅层评估：最多检查当前玩家同回合的下一动作，不替对手滚到终局，也不会为当前结论占用 GPU。".to_string(),
            );
        } else {
            paragraphs.push(format!(
                "一条随机样例续弈最终得到 {:?} VP，官方胜者为玩家 {:?}；这只是样例，不是主变化。",
                candidate.sample_random_continuation.final_victory_points,
                candidate.sample_random_continuation.official_winners
            ));
        }
    }

    let mut evidence = vec![
        format!("动作：{}", serialized.summary),
        if uses_rule {
            format!(
                "规则评分：{:+.2}；相对偏好：{}；已评估全部 {} 个合法动作",
                candidate
                    .rule_score
                    .unwrap_or(candidate.average_victory_point_margin),
                win_rate,
                report.evaluated_action_count,
            )
        } else {
            format!(
                "访问：{}；访问占比：{}；共享胜分率：{}",
                visits, visit_share, win_rate
            )
        },
        match candidate.average_final_victory_points {
            Some(victory_points) if uses_rule => format!(
                "动作后当前 VP：{victory_points:.1}；当前 VP 分差：{:+.1}",
                candidate.average_victory_point_margin
            ),
            Some(victory_points) => format!(
                "平均终局 VP：{victory_points:.1}；平均 VP 分差：{:+.1}",
                candidate.average_victory_point_margin
            ),
            None if uses_neural_tree => format!(
                "多层树回传 VP 分差估计：{:+.1}",
                candidate.average_victory_point_margin
            ),
            None if uses_successor_model => format!(
                "后继模型 VP 分差估计：{:+.1}",
                candidate.average_victory_point_margin
            ),
            None => format!(
                "VP 分差估计：{:+.1}",
                candidate.average_victory_point_margin
            ),
        },
        format!("标准误：{}", standard_error),
    ];
    if !effect.is_empty() {
        evidence.push(format!("立即变化：{}", effect));
    }
    if let Some(prior) = policy_prior {
        evidence.push(if uses_rule {
            format!("规则相对偏好：{}", prior)
        } else {
            format!("策略网络先验：{}", prior)
        });
    }
    if let Some(breakdown) = candidate.rule_score_breakdown {
        evidence.push(format!(
            "规则分量：即时 VP {:+.2}，潜在 VP {:+.2}，收入 {:+.2}，现金 {:+.2}，产业 {:+.2}，网络 {:+.2}，资源 {:+.2}，安全 {:+.2}，弃牌 {:+.2}，下一动作 {:+.2}",
            breakdown.immediate_vp,
            breakdown.potential_vp,
            breakdown.income,
            breakdown.cash,
            breakdown.industry,
            breakdown.network,
            breakdown.resources,
            breakdown.safety,
            breakdown.card_value,
            breakdown.lookahead,
        ));
    }

    Ok(ExplanationJson {
        revision,
        action_key: action_key.to_string(),
        question: question.trim().to_string(),
        answer: paragraphs.join("\n\n"),
        evidence,
        caveat: if report.method == BATCHED_NEURAL_PUCT_METHOD {
            format!(
                "当前是多确定化、多层神经 PUCT：实际最大深度 {}，完成 {} 个神经叶估值和 {} 个推理批次。隐藏信息确定化与模型本身仍是近似，数值尚未经过实战校准。",
                report.max_search_depth.unwrap_or(0),
                report.neural_leaf_evaluations.unwrap_or(0),
                report.inference_batches.unwrap_or(0),
            )
        } else if report.method == RULE_ECONOMIC_METHOD {
            "产业兑现概率和未来资源需求是启发式估计，评分不是胜率；当前使用已知手牌和公开棋盘，没有展开完整的对手应对与终局搜索。"
                .to_string()
        } else if report.method == RULE_DECISION_TREE_METHOD {
            "当前是 CPU 规则决策树：每个合法动作只评估一次前后状态，综合 VP、收入、现金、产业、网络、资源和安全分量；没有终局续弈、神经网络或胜率校准。"
                .to_string()
        } else if report.method == ROOT_PUCT_ACTION_VALUE_METHOD {
            "当前是策略先验 + 根节点 PUCT + 批量一步后继价值；价值已参与排序，但仍不是完整深层神经树搜索，也尚未校准。"
                .to_string()
        } else if report.method == ROOT_PUCT_METHOD {
            "当前使用已训练策略先验引导根节点 PUCT，价值统计仍来自随机续弈，且尚未经过胜率校准。"
                .to_string()
        } else {
            "当前是确定化根节点 UCB + 随机续弈基线，不是已训练策略，也没有经过胜率校准。"
                .to_string()
        },
    })
}

fn serialize_candidate(
    candidate: &RootActionEstimate,
    runner: &GameRunner,
    root_player: usize,
    reveal_private_cards: bool,
) -> AnalysisCandidateJson {
    let (summary, selections) = describe_action(
        &candidate.action.intent,
        runner,
        root_player,
        reveal_private_cards,
    );
    let effect = &candidate.immediate_effect;
    AnalysisCandidateJson {
        rank: candidate.rank,
        action_key: candidate.action_key.clone(),
        action_type: action_type_code(candidate.action.intent.action_type),
        action_label: action_type_label(
            candidate.action.intent.action_type,
            runner.framework.board.state.era,
        ),
        summary,
        selections,
        visits: candidate.visits,
        visit_share: candidate.visit_share,
        value_source: candidate.value_source.clone(),
        value_sample_count: candidate.value_sample_count,
        estimated_shared_win_rate: candidate.estimated_shared_win_rate,
        estimated_outright_win_rate: candidate.estimated_outright_win_rate,
        estimated_tied_first_rate: candidate.estimated_tied_first_rate,
        average_final_victory_points: candidate.average_final_victory_points,
        average_victory_point_margin: candidate.average_victory_point_margin,
        shared_win_rate_standard_error: candidate.shared_win_rate_standard_error,
        policy_probability: candidate.policy_probability,
        calibrated_win_rate: candidate.calibrated_win_rate,
        rule_score: candidate.rule_score,
        rule_score_breakdown: candidate.rule_score_breakdown,
        immediate_effect: ImmediateEffectJson {
            money_delta: effect.player_money_delta[root_player],
            income_level_delta: effect.player_income_level_delta[root_player],
            victory_points_delta: effect.player_victory_points_delta[root_player],
            visible_victory_points_delta: effect.player_visible_victory_points_delta[root_player],
            potential_era_victory_points_delta: effect.player_potential_era_victory_points_delta
                [root_player],
            hand_size_delta: effect.player_hand_size_delta[root_player],
            round_spend_delta: effect.player_round_spend_delta[root_player],
            roads_on_board_delta: effect.roads_on_board_delta,
            buildings_on_board_delta: effect.buildings_on_board_delta,
            flipped_buildings_delta: effect.flipped_buildings_delta,
            draw_deck_size_delta: effect.draw_deck_size_delta,
            market_coal_delta: effect.market_coal_delta,
            market_iron_delta: effect.market_iron_delta,
            wild_location_pool_delta: effect.wild_location_pool_delta,
            wild_industry_pool_delta: effect.wild_industry_pool_delta,
            turn_advanced: effect.turn_count_delta > 0,
            phase_changed: effect.phase_before != effect.phase_after
                || effect.era_before != effect.era_after,
        },
        highlights: highlights_for_intent(&candidate.action.intent),
        sample_random_continuation: SampleContinuationJson {
            label: if candidate.value_source == RULE_IMMEDIATE_SCORE_SOURCE {
                "浅层规则评分（同回合前瞻，无终局续弈）"
            } else {
                "随机样例续弈（非主变化）"
            },
            steps: candidate
                .sample_random_continuation
                .steps
                .iter()
                .map(|step| SampleContinuationStepJson {
                    action_number: step.action_number,
                    player_idx: step.player_idx,
                    action_type: action_type_code(step.action_type),
                    action_label: action_type_label(
                        step.action_type,
                        runner.framework.board.state.era,
                    ),
                    action_key: step.action_key.clone(),
                })
                .collect(),
            final_victory_points: candidate
                .sample_random_continuation
                .final_victory_points
                .clone(),
            official_winners: candidate
                .sample_random_continuation
                .official_winners
                .clone(),
        },
    }
}

fn describe_action(
    intent: &ActionIntent,
    runner: &GameRunner,
    player_idx: usize,
    reveal_private_cards: bool,
) -> (String, Vec<String>) {
    let mut selections = Vec::new();
    if let Some(industry) = intent.selected_industry {
        selections.push(industry_str(industry).to_string());
    }
    if let Some(industry) = intent.selected_second_industry {
        if Some(industry) != intent.selected_industry {
            selections.push(format!("再研发 {}", industry_str(industry)));
        }
    }
    if reveal_private_cards {
        if let Some(card_idx) = intent.selected_card_idx {
            selections.push(format!("弃 {}", card_label(runner, player_idx, card_idx)));
        }
        if intent.action_type == ActionType::Scout {
            for card_idx in &intent.scout_additional_discard_indices {
                selections.push(format!("弃 {}", card_label(runner, player_idx, *card_idx)));
            }
        }
    } else {
        let hidden_discard_count = usize::from(intent.selected_card_idx.is_some())
            + intent.scout_additional_discard_indices.len();
        if hidden_discard_count > 0 {
            selections.push(format!("弃 {hidden_discard_count} 张手牌（暗牌）"));
        }
    }
    if let Some(location) = intent.selected_build_location {
        selections.push(format!("{} #{}", town_name_for_bl(location), location));
    }
    if let Some(mode) = intent.selected_network_mode {
        selections.push(match mode {
            NetworkMode::Single => "单线".to_string(),
            NetworkMode::Double => "双线".to_string(),
        });
    }
    if let Some(road_idx) = intent.selected_road_idx {
        selections.push(crate::core::static_data::road_label(road_idx));
    }
    if let Some(road_idx) = intent.selected_second_road_idx {
        selections.push(crate::core::static_data::road_label(road_idx));
    }
    for source in &intent.chosen_coal_sources {
        selections.push(resource_source_label("煤", *source));
    }
    for source in &intent.chosen_iron_sources {
        selections.push(resource_source_label("铁", *source));
    }
    for source in &intent.chosen_beer_sources {
        selections.push(beer_source_label(*source));
    }
    if let Some(source) = intent.chosen_action_beer_source {
        selections.push(action_beer_source_label(source));
    }
    for sell in &intent.sell_choices {
        selections.push(format!(
            "出售 {} #{}",
            town_name_for_bl(sell.location),
            sell.location
        ));
    }
    if intent.action_type == ActionType::Sell {
        if let Some(industry) = intent.free_development_choice {
            selections.push(format!("免费研发 {}", industry_str(industry)));
        }
    }

    let label = action_type_label(intent.action_type, runner.framework.board.state.era);
    let summary = if selections.is_empty() {
        label.to_string()
    } else {
        format!("{} · {}", label, selections.join(" · "))
    };
    (summary, selections)
}

fn card_label(runner: &GameRunner, player_idx: usize, card_idx: usize) -> String {
    runner
        .framework
        .board
        .state
        .players
        .get(player_idx)
        .and_then(|player| player.hand.cards.get(card_idx))
        .map(|card: &Card| format_card_label(&card.card_type))
        .unwrap_or_else(|| format!("手牌 #{}", card_idx))
}

fn resource_source_label(resource: &str, source: ResourceSource) -> String {
    match source {
        ResourceSource::Building(location) => {
            format!("{}：{} #{}", resource, town_name_for_bl(location), location)
        }
        ResourceSource::Market => format!("{}：市场", resource),
    }
}

fn beer_source_label(source: BeerSellSource) -> String {
    match source {
        BeerSellSource::Building(location) => {
            format!("啤酒：{} #{}", town_name_for_bl(location), location)
        }
        BeerSellSource::TradePost(slot) => format!("啤酒：商人位 #{}", slot),
    }
}

fn action_beer_source_label(source: BreweryBeerSource) -> String {
    match source {
        BreweryBeerSource::OwnBrewery(location) => {
            format!("自有啤酒：{} #{}", town_name_for_bl(location), location)
        }
        BreweryBeerSource::OpponentBrewery(location) => {
            format!("他人啤酒：{} #{}", town_name_for_bl(location), location)
        }
    }
}

fn highlights_for_intent(intent: &ActionIntent) -> AnalysisHighlightsJson {
    let mut highlights = AnalysisHighlightsJson::default();
    if let Some(location) = intent.selected_build_location {
        highlights.building_locations.push(location);
    }
    if let Some(road) = intent.selected_road_idx {
        highlights.roads.push(road);
    }
    if let Some(road) = intent.selected_second_road_idx {
        highlights.roads.push(road);
    }
    for source in intent
        .chosen_coal_sources
        .iter()
        .chain(&intent.chosen_iron_sources)
    {
        if let ResourceSource::Building(location) = source {
            highlights.source_locations.push(*location);
        }
    }
    for source in &intent.chosen_beer_sources {
        match source {
            BeerSellSource::Building(location) => highlights.source_locations.push(*location),
            BeerSellSource::TradePost(slot) => highlights.merchant_slots.push(*slot),
        }
    }
    if let Some(source) = intent.chosen_action_beer_source {
        let location = match source {
            BreweryBeerSource::OwnBrewery(location)
            | BreweryBeerSource::OpponentBrewery(location) => location,
        };
        highlights.source_locations.push(location);
    }
    for sell in &intent.sell_choices {
        highlights.building_locations.push(sell.location);
    }
    highlights.building_locations.sort_unstable();
    highlights.building_locations.dedup();
    highlights.source_locations.sort_unstable();
    highlights.source_locations.dedup();
    highlights.roads.sort_unstable();
    highlights.roads.dedup();
    highlights.merchant_slots.sort_unstable();
    highlights.merchant_slots.dedup();
    highlights
}

fn effect_sentence(effect: &ImmediateEffectJson) -> String {
    let mut effects = Vec::new();
    if effect.money_delta != 0 {
        effects.push(format!("现金 {:+}", effect.money_delta));
    }
    if effect.income_level_delta != 0 {
        effects.push(format!("收入轨 {:+}", effect.income_level_delta));
    }
    if effect.potential_era_victory_points_delta != 0 {
        effects.push(format!(
            "本时代潜在 VP {:+}",
            effect.potential_era_victory_points_delta
        ));
    }
    if effect.roads_on_board_delta != 0 {
        effects.push(format!("线路 {:+}", effect.roads_on_board_delta));
    }
    if effect.buildings_on_board_delta != 0 {
        effects.push(format!("建筑 {:+}", effect.buildings_on_board_delta));
    }
    if effect.flipped_buildings_delta != 0 {
        effects.push(format!("翻面建筑 {:+}", effect.flipped_buildings_delta));
    }
    if effect.wild_location_pool_delta != 0 || effect.wild_industry_pool_delta != 0 {
        effects.push("获得万能牌".to_string());
    }
    if effects.is_empty() {
        "立即公共分数与资源没有显著变化。".to_string()
    } else {
        format!("立即变化为 {}。", effects.join("、"))
    }
}

fn risk_sentence(candidate: &RootActionEstimate, report: &RootSearchReport) -> String {
    let uncertainty = candidate
        .shared_win_rate_standard_error
        .map(format_percent)
        .unwrap_or_else(|| "无法估计".to_string());
    if candidate.value_source == RULE_IMMEDIATE_SCORE_SOURCE {
        format!(
            "主要风险是规则树只看当前动作和同回合一层前瞻，没有显式建模对手应对、隐藏牌分布或终局兑现；{:+.2} 的相对评分不能直接解释为胜率。",
            candidate
                .rule_score
                .unwrap_or(candidate.average_victory_point_margin)
        )
    } else if candidate.value_source == NEURAL_TREE_VALUE_SOURCE {
        format!(
            "主要风险是当前这步只有 {} 次树搜索价值回传，标准误为 {}，整次搜索实际最大深度为 {}；隐藏牌确定化抽样、叶节点模型误差和更深层应对仍会影响结果。",
            candidate.value_sample_count,
            uncertainty,
            report.max_search_depth.unwrap_or(0),
        )
    } else if candidate.value_source == SUCCESSOR_MODEL_VALUE_SOURCE {
        format!(
            "主要风险是当前只有 {} 个隐藏牌确定化后继样本，模型估计标准误为 {}；搜索只评估一步后继局面，模型误差、隐藏牌抽样和对手后续的高水平应对仍会影响结果。",
            candidate.value_sample_count, uncertainty
        )
    } else {
        format!(
            "主要风险是当前只有 {} 次访问，胜分率标准误为 {}，且后续玩家使用随机策略；资源时机、对手阻断和高水平应对尚未被可靠建模。",
            candidate.visits, uncertainty
        )
    }
}

fn format_percent(value: f64) -> String {
    let rounded_tenths = (value * 1000.0).round();
    format!("{:.1}%", rounded_tenths / 10.0)
}

fn choice_ranks_mentioned(question: &str) -> Vec<usize> {
    [
        (1, ["一选", "第一", "top 1", "top1"]),
        (2, ["二选", "第二", "top 2", "top2"]),
        (3, ["三选", "第三", "top 3", "top3"]),
    ]
    .into_iter()
    .filter_map(|(rank, markers)| {
        markers
            .iter()
            .any(|marker| question.contains(marker))
            .then_some(rank)
    })
    .collect()
}

fn choice_rank_label(rank: usize) -> String {
    match rank {
        1 => "一选".to_string(),
        2 => "二选".to_string(),
        3 => "三选".to_string(),
        _ => format!("第 {rank} 选择"),
    }
}

fn action_type_code(action_type: ActionType) -> &'static str {
    match action_type {
        ActionType::BuildBuilding => "build",
        ActionType::BuildRailroad => "network",
        ActionType::BuildDoubleRailroad => "double_network",
        ActionType::Develop => "develop",
        ActionType::DevelopDouble => "develop_double",
        ActionType::Sell => "sell",
        ActionType::Loan => "loan",
        ActionType::Scout => "scout",
        ActionType::Pass => "pass",
    }
}

fn action_type_label(action_type: ActionType, era: Era) -> &'static str {
    match action_type {
        ActionType::BuildBuilding => "建造",
        ActionType::BuildRailroad => match era {
            Era::Canal => "铺设运河",
            Era::Railroad => "铺设铁路",
        },
        ActionType::BuildDoubleRailroad => "铺设双铁路",
        ActionType::Develop => "研发",
        ActionType::DevelopDouble => "双研发",
        ActionType::Sell => "出售",
        ActionType::Loan => "贷款",
        ActionType::Scout => "侦察",
        ActionType::Pass => "跳过",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::legal_actions::enumerate_legal_actions;
    use crate::game::search::{search_top_actions, RootSearchConfig};

    #[test]
    fn percentages_use_the_same_half_up_rounding_as_the_browser() {
        assert_eq!(format_percent(162.0 / 800.0), "20.3%");
    }

    #[test]
    fn economic_rule_report_keeps_its_identity_and_explanation() {
        let runner = GameRunner::new(2, Some(8_200));
        let report = crate::game::rule_ai::rule_decision_report(
            &runner,
            &crate::game::rule_ai::RuleDecisionConfig::default(),
        )
        .unwrap();
        let serialized = serialize_analysis(&report, &runner, 1, 1);
        assert_eq!(serialized.method, RULE_ECONOMIC_METHOD);
        assert!(serialized.method_label.contains("经济规划 v2"));
        assert!(serialized.recommendations[0].rule_score.is_some());
        let explanation = explain_analysis_question(
            &report,
            &runner,
            1,
            &report.recommendations[0].action_key,
            "选择依据",
        )
        .unwrap();
        assert!(explanation.answer.contains("经济评分"));
        assert!(explanation.caveat.contains("启发式"));
    }

    #[test]
    fn top_candidate_can_be_compared_with_the_second_choice_in_chinese() {
        let runner = GameRunner::new(2, Some(8_201));
        let report = search_top_actions(
            &runner,
            &RootSearchConfig {
                simulations: 3,
                recommendation_count: 3,
                max_rollout_actions: 160,
                sample_continuation_length: 1,
                seed: 8_202,
                ..RootSearchConfig::default()
            },
        )
        .expect("search should return three recommendations");
        assert_eq!(report.recommendations.len(), 3);
        let top = &report.recommendations[0];
        let second_summary = describe_action(
            &report.recommendations[1].action.intent,
            &runner,
            report.root_player,
            true,
        )
        .0;

        let explanation = explain_analysis_question(
            &report,
            &runner,
            1,
            &top.action_key,
            "为什么这步比第二选择好？",
        )
        .expect("comparison explanation should serialize");

        assert!(explanation.answer.contains("与二选"));
        assert!(explanation.answer.contains(&second_summary));
        assert!(explanation.answer.contains("当前排序先按搜索访问数"));
        assert!(explanation.answer.contains("二选的即时结果"));
    }

    #[test]
    fn observer_analysis_redacts_the_acting_players_card_identity() {
        let runner = GameRunner::new(2, Some(8_203));
        let action = enumerate_legal_actions(&runner)
            .expect("initial actions should enumerate")
            .into_iter()
            .find(|action| action.intent.selected_card_idx.is_some())
            .expect("an initial legal action should discard a card");
        let card_idx = action.intent.selected_card_idx.unwrap();
        let card_name = format_card_label(
            &runner.framework.board.state.players[runner.framework.current_player]
                .hand
                .cards[card_idx]
                .card_type,
        );

        let (_, visible_selections) = describe_action(
            &action.intent,
            &runner,
            runner.framework.current_player,
            true,
        );
        let (_, redacted_selections) = describe_action(
            &action.intent,
            &runner,
            runner.framework.current_player,
            false,
        );

        let private_discard = format!("弃 {card_name}");
        assert!(visible_selections.contains(&private_discard));
        assert!(redacted_selections
            .iter()
            .any(|selection| selection.contains("手牌（暗牌）")));
        assert!(!redacted_selections.contains(&private_discard));
    }

    #[test]
    fn successor_value_explanation_describes_model_risk_and_feedback() {
        let runner = GameRunner::new(2, Some(8_301));
        let mut report = search_top_actions(
            &runner,
            &RootSearchConfig {
                simulations: 3,
                recommendation_count: 3,
                max_rollout_actions: 160,
                sample_continuation_length: 1,
                seed: 8_302,
                ..RootSearchConfig::default()
            },
        )
        .expect("search should return three recommendations");
        report.method = ROOT_PUCT_ACTION_VALUE_METHOD.to_string();
        report.value_source = SUCCESSOR_MODEL_VALUE_SOURCE.to_string();
        for candidate in &mut report.recommendations {
            candidate.value_source = SUCCESSOR_MODEL_VALUE_SOURCE.to_string();
            candidate.value_sample_count = 4;
            candidate.policy_probability = Some(1.0 / 3.0);
        }
        let top_action_key = report.recommendations[0].action_key.clone();

        let explanation = explain_analysis_question(
            &report,
            &runner,
            1,
            &top_action_key,
            "为什么一选比二选好？这步的风险在哪里？",
        )
        .expect("successor-value explanation should serialize");

        assert!(explanation.answer.contains("后继价值反馈"));
        assert!(explanation.answer.contains("4 个隐藏牌确定化后继样本"));
        assert!(explanation.answer.contains("搜索只评估一步后继局面"));
        assert!(!explanation.answer.contains("后续玩家使用随机策略"));
    }

    #[test]
    fn neural_tree_explanation_reports_real_depth_and_leaf_feedback() {
        let runner = GameRunner::new(2, Some(8_401));
        let mut report = search_top_actions(
            &runner,
            &RootSearchConfig {
                simulations: 3,
                recommendation_count: 3,
                max_rollout_actions: 160,
                sample_continuation_length: 1,
                seed: 8_402,
                ..RootSearchConfig::default()
            },
        )
        .expect("search should return three recommendations");
        report.method = BATCHED_NEURAL_PUCT_METHOD.to_string();
        report.value_source = NEURAL_TREE_VALUE_SOURCE.to_string();
        report.max_search_depth = Some(4);
        report.neural_leaf_evaluations = Some(37);
        report.inference_batches = Some(3);
        for candidate in &mut report.recommendations {
            candidate.value_source = NEURAL_TREE_VALUE_SOURCE.to_string();
            candidate.value_sample_count = candidate.visits;
            candidate.policy_probability = Some(1.0 / 3.0);
        }
        let top_action_key = report.recommendations[0].action_key.clone();

        let serialized = serialize_analysis(&report, &runner, 1, 25);
        assert_eq!(
            serialized.method_label,
            "策略价值网络 · 多层 PUCT · 隐藏牌确定化"
        );
        assert_eq!(serialized.max_search_depth, Some(4));
        assert_eq!(serialized.neural_leaf_evaluations, Some(37));
        assert_eq!(serialized.inference_batches, Some(3));

        let explanation = explain_analysis_question(
            &report,
            &runner,
            1,
            &top_action_key,
            "为什么一选比二选好？这个胜率可靠吗，有什么风险？",
        )
        .expect("neural-tree explanation should serialize");

        assert!(explanation
            .answer
            .contains("多层树中的对手应对与叶节点价值反馈"));
        assert!(explanation.answer.contains("当前实际最大深度为 4"));
        assert!(explanation.answer.contains("共完成 37 个神经叶估值"));
        assert!(explanation.caveat.contains("3 个推理批次"));
        assert!(!explanation.answer.contains("搜索只评估一步后继局面"));
        assert!(!explanation.answer.contains("后续玩家使用随机策略"));
    }
}

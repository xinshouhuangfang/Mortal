use riichi::agent::{BatchAgent, HumanAgent, Tsumogiri};
use riichi::arena::game::{BatchGame, Index};
use std::io::{self, Write};

fn main() -> anyhow::Result<()> {
    println!("===== Mortal 麻将 — 人机对战 =====");
    println!("你将与 Tsumogiri AI 对战（AI 永远摸切）");
    println!("每轮你会看到手牌和可用操作，输入指令后按回车。");
    println!("可用指令:");
    println!("  牌名(如 1m, 5p, E) 或 数字 0-36: 打牌");
    println!("  riichi: 立直");
    println!("  tsumo:  自摸和");
    println!("  ron:    荣和");
    println!("  chi_l / chi_m / chi_h: 吃（左/中/右）");
    println!("  pon:    碰");
    println!("  daiminkan / kakan / ankan: 杠");
    println!("  ryukyoku: 流局");
    println!("  pass / none: 跳过");
    println!();

    let games: u64 = loop {
        print!("请输入对局数 (1-N): ");
        io::stdout().flush()?;
        let mut line = String::new();
        io::stdin().read_line(&mut line)?;
        match line.trim().parse() {
            Ok(n) if n >= 1 => break n,
            _ => println!("请输入正整数"),
        }
    };

    println!("游戏开始！\n");

    let g = BatchGame::tenhou_hanchan(true);
    let mut agents: Vec<Box<dyn BatchAgent>> = vec![
        Box::new(HumanAgent::new_batched(&[0, 1, 2, 3])?),
        Box::new(Tsumogiri::new_batched(&[3, 2, 1, 0])?),
    ];

    let indexes: Vec<_> = (0..games)
        .flat_map(|_| {
            [
                [
                    Index { agent_idx: 0, player_id_idx: 0 },
                    Index { agent_idx: 0, player_id_idx: 1 },
                    Index { agent_idx: 1, player_id_idx: 1 },
                    Index { agent_idx: 1, player_id_idx: 0 },
                ],
                [
                    Index { agent_idx: 1, player_id_idx: 3 },
                    Index { agent_idx: 1, player_id_idx: 2 },
                    Index { agent_idx: 0, player_id_idx: 2 },
                    Index { agent_idx: 0, player_id_idx: 3 },
                ],
            ]
        })
        .collect();

    let seeds: Vec<_> = (0..games)
        .flat_map(|i| [(60000 + i, 0x5678), (60000 + i, 0x5678)])
        .collect();

    let results = g.run(&mut agents, &indexes, &seeds)?;

    println!("\n===== 游戏结束 =====");
    for (i, result) in results.iter().enumerate() {
        let rank = result.rankings().rank_by_player[i % 4];
        let pt = match rank {
            0 => "一位 (+6)",
            1 => "二位 (+4)",
            2 => "三位 (+2)",
            _ => "四位 (0)",
        };
        println!("  半庄 {}: 第{}位 {}", i + 1, rank + 1, pt);
    }

    Ok(())
}

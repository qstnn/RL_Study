"""创建场景前固定评估模板；避免先挂背载再移除的状态路径依赖。"""
def evaluation_template(cfg, args):
    if args["mode"] == "evaluate":
        # 与15/18冻结基准的创建路径一致；evaluate随后写每个case的真实坡度和质量。
        cfg.fixed_slope_deg=25.
        cfg.fixed_payload_kg=0.
        cfg.fixed_towed_kg=40.
        cfg.fixed_speed_mps=0.
    elif args["mode"] == "play":
        cfg.fixed_slope_deg=args["slope_deg"]
        cfg.fixed_payload_kg=args["payload_kg"]
        cfg.fixed_towed_kg=args["towed_kg"]
        cfg.fixed_speed_mps=args["speed"]
    else:
        raise ValueError("Only frozen evaluation/play use a fixed scene template")

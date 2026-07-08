def register(options_templates, options_section, OptionInfo):
    from modules.ui_components import FormColorPicker

    options_templates.update(
        options_section(
            (None, "Forge Hidden Options"),
            {
                "VERSION_UID": OptionInfo(None, "internal version for breaking-changes"),
                "forge_preset": OptionInfo("sd"),
                "forge_additional_modules": OptionInfo([]),
                "forge_unet_storage_dtype": OptionInfo("Automatic"),
            },
        )
    )
    options_templates.update(
        options_section(
            ("ui_forgecanvas", "Forge Canvas", "ui"),
            {
                "forge_canvas_height": OptionInfo(512, "Canvas Height").info("in pixels").needs_reload_ui(),
                "forge_canvas_toolbar_always": OptionInfo(False, "Always Visible Toolbar").info("disabled: toolbar only appears when hovering the canvas").needs_reload_ui(),
                "forge_canvas_consistent_brush": OptionInfo(False, "Fixed Brush Size").info("disabled: the brush size is <b>pixel-space</b>, the brush stays small when zoomed out ; enabled: the brush size is <b>canvas-space</b>, the brush stays big when zoomed in").needs_reload_ui(),
                "forge_canvas_plain": OptionInfo(False, "Plain Background").info("disabled: checkerboard pattern ; enabled: solid color").needs_reload_ui(),
                "forge_canvas_plain_color": OptionInfo("#808080", "Solid Color for Plain Background", FormColorPicker, {}).needs_reload_ui(),
            },
        )
    )

    import gradio as gr
    options_templates.update(
        options_section(
            ("ui_anima_qwen", "Anima Qwen 3.5 Options", "ui"),
            {
                "anima_qwen35_use_calibration": OptionInfo(False, "Anima Qwen 3.5: Use Calibration").info("Apply per-dimension affine calibration. Recommended: OFF if using Alignment."),
                "anima_qwen35_use_alignment": OptionInfo(True, "Anima Qwen 3.5: Use Concept Alignment").info("Apply Procrustes rotation to align 4B spatial/pose concept directions with 0.6B."),
                "anima_qwen35_alignment_strength": OptionInfo(0.5, "Anima Qwen 3.5: Alignment Strength", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.05}).info("Blend distribution center: 0=keep 4B's own scale, 1=shift to 0.6B's scale. 0.5 is recommended."),
                "anima_qwen35_output_scale": OptionInfo(1.0, "Anima Qwen 3.5: Output Scale", gr.Slider, {"minimum": 0.1, "maximum": 10.0, "step": 0.1}).info("Additional uniform scaling factor applied at the end."),
            },
        )
    )


Nikemon automatic rig-prep package

Generated files:
- nikemon_character_animator_layers.psd: layered PSD generated automatically with Photoshop.
- full_body.png: transparent full-body cutout.
- *.png: automatically segmented body parts.
- manifest.json: layer positions and z-order.
- build_layered_psd_direct.jsx: Photoshop script used to assemble the layered PSD.
- contact_sheet.png: visual check sheet of generated parts.

Recommended workflow:
1. Install/open Adobe Character Animator from Creative Cloud if it is not installed.
2. Import nikemon_character_animator_layers.psd.
3. Let Character Animator detect the puppet structure.
4. Refine only if needed: pins/handles around shoulders, hands, hips, and feet.
5. Record or apply body/arm motion, then export a transparent-background video.
6. Composite that render with the original video/audio.

This is an automatic first pass. The PSD is already generated, so Photoshop scripting is no longer required unless you regenerate parts.

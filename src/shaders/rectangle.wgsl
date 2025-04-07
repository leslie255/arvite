@group(0) @binding(0) var<uniform> fill_color: vec4<f32>;
@group(0) @binding(1) var<uniform> model_view: mat4x4<f32>;
@group(0) @binding(2) var<uniform> projection: mat4x4<f32>;

@vertex
fn vs_main(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
    return projection * model_view * vec4<f32>(position.xy, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return fill_color;
}

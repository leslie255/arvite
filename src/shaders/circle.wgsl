@group(0) @binding(0) var<uniform> model_view: mat4x4<f32>;
@group(0) @binding(1) var<uniform> projection: mat4x4<f32>;
@group(0) @binding(2) var<uniform> fill_color: vec4<f32>;
@group(0) @binding(3) var<uniform> inner_radius: f32;

fn sd_circle(p: vec2<f32>, r: f32) -> f32 {
    return length(p) - r;
}

struct VertexOutput {
    @location(0) uv: vec2<f32>,
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@location(0) position: vec2<f32>, @location(1) uv: vec2<f32>) -> VertexOutput {
    var result: VertexOutput;
    result.uv = uv;
    result.position = projection * model_view * vec4<f32>(position.xy, 0.0, 1.0);
    return result;
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let aaf = length(fwidth(vertex.uv));
    let sd_outer = sd_circle(vertex.uv, 1.0);
    let sd_inner = sd_circle(vertex.uv, inner_radius);
    let sd = max(sd_outer, -sd_inner);
    let brightness = smoothstep(-aaf * 0.5, aaf * 0.5, -sd);
    return brightness * fill_color;
}

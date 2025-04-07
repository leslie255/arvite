@group(0) @binding(0) var<uniform> model_view: mat4x4<f32>;
@group(0) @binding(1) var<uniform> projection: mat4x4<f32>;
@group(0) @binding(2) var<uniform> gamma: f32;

@group(0) @binding(3) var the_texture: texture_2d<f32>;
@group(0) @binding(4) var the_sampler: sampler;

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
    var sample = textureSample(the_texture, the_sampler, vertex.uv);
    sample = pow(sample, vec4<f32>(gamma));
    return sample;
}

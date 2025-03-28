#version 140

in vec2 position;
in vec2 uv;

out vec2 vert_position;
out vec2 vert_uv;

uniform mat4 model_view;
uniform mat4 projection;

void main() {
    gl_Position = projection * model_view * vec4(position.xy, 0.0, 1.0);
    vert_position = position;
    vert_uv = uv;
}

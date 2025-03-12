#version 140

in vec2 vert_normalized;

out vec4 frag_color;

uniform float inner_radius_normalized;
uniform float outer_radius_normalized;
uniform vec4 color;

void main() {
    float r_in2 = inner_radius_normalized * inner_radius_normalized;
    float r_out2 = outer_radius_normalized * outer_radius_normalized;
    float r2 = vert_normalized.x * vert_normalized.x + vert_normalized.y * vert_normalized.y;
    if (r2 < r_in2 || r2 > r_out2) {
        discard;
    }
    frag_color = color;
}

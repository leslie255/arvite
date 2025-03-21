#version 140

in vec2 vert_position;

out vec4 frag_color;

// SDF of a line segment.
float sd_segment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p-a;
    vec2 ba = b-a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

void main() {
    vec2 pa = vec2(100.0, 100.0);
    vec2 pb = vec2(200.0, 200.0);
    float sd = 1.0 - sd_segment(vert_position, pa, pb);
    frag_color = sd * vec4(1., 1., 1., 1.);
}

#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 colour;

layout(binding = 0) uniform UboViewProjection {
    mat4 projection;
    mat4 view;
} vp;

layout(push_constant) uniform PushModel {
    mat4 model;
} m;

layout(location = 0) out vec3 fragColour;

void main() {
    gl_Position = vp.projection * vp.view * m.model * vec4(position, 1.0);
    fragColour = colour;
}
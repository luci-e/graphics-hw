#pragma once

#include <thread>
#include <iostream>
#include "yocto_bvh.h"
#include "yocto_img.h"
#include "yocto_math.h"
#include "yocto_obj.h"
#include "yocto_utils.h"
#include <math.h>
#include <omp.h>
#include "embree2\rtcore.h"
#include "embree2\rtcore_ray.h"
#include "yobree.h"

#define MAX_BOUNCES 1

namespace rt {

	template <typename T>
	const void print_vector(T vector, std::string prefix, int size);

	ybvh::scene* make_bvh(yobj::scene* scn);

	// Take a ray and a camera, get the pixels
	//ym::vec2f reverse_eval(const yobj::camera* cam, const ym::ray3f ray);

	ym::ray3f eval_camera(const yobj::camera* cam, const ym::vec2f& uv);

	// ---------------------------------------------------------------------------------------------------- //
	// Should I manually copy and paste from the slides, possibly introducing errors or just use a pre-made
	// function that I know works ? To be fair though, a different interpolation algorithm is provided

	// Hold the coordinates of a texture pixel and a weight attached to it
	struct neighbour_pixel {
		ym::vec2i px_coords;
		float weight;
	};

	// Round to closest number, snap to grid
	float round_and_snap(float x);

	void closest_neighbours_pixel(neighbour_pixel * neighbours, ym::vec2f st, int width, int height);

	ym::vec4f eval_texture2(const yobj::texture* txt, const ym::vec2f& texcoord, bool srgb);

	// It's got asserts, it's what functions crave.
	inline ym::vec4f lookup_texture(const yobj::texture* txt, const ym::vec2i& ij, bool srgb);

	ym::vec4f eval_texture(const yobj::texture* txt, const ym::vec2f& texcoord, bool srgb);

	// ---------------------------------------------------------------------------------------------------- //


	// Use this one for 3-dimensional vectors
	ym::vec3f barycentric_to_vec(ym::vec4f bar_coord, std::vector<ym::vec3f> vecs, int vec_no);

	// Use this one 2 dimensional vectors ( for textures )
	ym::vec2f barycentric_to_vec(ym::vec4f bar_coord, std::vector<ym::vec2f> vecs, int vec_no);

	// Compute the opaqueness at the intersection point
	float opaqueness_at_intersection(const yobj::scene* scn, const ybvh::scene* bvh, const ybvh::intersection_point intersection);

	struct point_info {
		ym::vec3f point;
		ym::vec3f normal;
		ym::vec2f texture_coords;
	};

	point_info get_point_info(const yobj::scene* scn, const ybvh::intersection_point intersection, bool is_textured);

	ym::vec4f shade_light(const yobj::scene* scn, const ybvh::scene* bvh, const yobj::instance* light, const int light_point, const ym::ray3f& ray, bool coloured_light);

	ym::vec4f shade(const yobj::scene* scn, const ybvh::scene* bvh,
					const std::vector<yobj::instance*>& lights, const ym::vec3f& amb,
					const ym::ray3f& ray, int ray_depth, bool coloured_light);

	ym::image4f raytrace(const yobj::scene* scn, const ybvh::scene* bvh,
						 const ym::vec3f& amb, int resolution, int samples, int cam_no, bool coloured_light);

	ym::image4f raytrace_mt(const yobj::scene* scn, const ybvh::scene* bvh,
							const ym::vec3f& amb, int resolution, int samples, int cam_no, bool coloured_light);

}
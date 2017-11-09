#pragma once
#include <iostream>
#include "yocto_bvh.h"
#include "yocto_img.h"
#include "yocto_math.h"
#include "yocto_obj.h"
#include "yocto_utils.h"
#include <math.h>
#include <omp.h>
#include "raytrace.h"
#include "embree2\rtcore.h"
#include "embree2\rtcore_ray.h"
#include "common\math\math.h"
#include <stdint.h>

namespace yobree {
	struct vertex { float x, y, z, a; };
	struct point { int32_t v0; };
	struct line { int32_t v0, v1; };
	struct triangle { int32_t v0, v1, v2; };

	struct flat4matrix {
		float matrix[16];

		float operator[](int i) const { return (matrix)[i]; }
		float & operator[](int i) { return (matrix)[i]; }
	};

	void copy_matrix(flat4matrix & matrix, ym::mat4f & ymatrix);

	struct bree_container {
		RTCDevice bree_device;
		RTCScene main_scene;
		std::vector< RTCScene > bree_scenes;
		std::vector<flat4matrix> instance_transformations;
	};

	struct point_info {
		ym::vec3f point;
		ym::vec3f normal;
		ym::vec2f texture_coords;
	};

	void print_error(std::string prefix, RTCError err_no);


	// ---------------------------------------------------------------------------------------------------- //
	// Should I manually copy and paste from the slides, possibly introducing errors or just use a pre-made
	// function that I know works ? To be fair though, a different interpolation algorithm is provided

	// It's got asserts, it's what functions crave.
	inline ym::vec4f lookup_texture(const yobj::texture* txt, const ym::vec2i& ij, bool srgb = true);

	ym::vec4f eval_texture(const yobj::texture* txt, const ym::vec2f& texcoord, bool srgb = true);

	// ---------------------------------------------------------------------------------------------------- //

	void import_yocto_meshes(yobj::scene * scn, bree_container & breec);

	// Compute the opaqueness at the intersection point
	float opaqueness_at_intersection(const yobj::scene* scn, RTCRay & intersection);

	point_info get_point_info(const yobj::scene* scn, RTCRay & intersection, bool is_textured = false);

	ym::vec4f shade_light(const yobj::scene* scn, bree_container & breec, const yobj::instance* light, const int light_point, RTCRay & ray, bool coloured_light = true);

	ym::vec4f shade(const yobj::scene * scn, bree_container & breec, const std::vector<yobj::instance*>& lights, const ym::vec3f & amb, RTCRay & ray, int ray_depth = 0, bool coloured_light = true);

	RTCRay eval_camera(const yobj::camera* cam, const ym::vec2f& uv);

	ym::image4f bree_raytrace(bree_container & breec, yobj::scene * scn, const ym::vec3f & amb, int resolution, int samples, int cam_no = 0, bool coloured_light = true);

}
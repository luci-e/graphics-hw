#include "yobree.h"

namespace yobree {

	void copy_matrix(flat4matrix & matrix, ym::mat4f & ymatrix) {
		for (int i = 0; i < 16; i++) {
			matrix[i] = ymatrix[i / 4][i % 4];
		}
	}


	void print_error(std::string prefix, RTCError err_no) {
		std::cout << prefix << std::endl;
		switch (err_no) {
		case RTC_NO_ERROR:
			std::cout << "All good!" << std::endl;
			break;
		case RTC_UNKNOWN_ERROR:
			std::cout << "Boh!" << std::endl;
			break;
		case RTC_INVALID_ARGUMENT:
			std::cout << "Bad arg!" << std::endl;
			break;
		case RTC_INVALID_OPERATION:
			std::cout << "Bad op!" << std::endl;
			break;
		case RTC_OUT_OF_MEMORY:
			std::cout << "Oh shit no mem!" << std::endl;
			break;
		case RTC_UNSUPPORTED_CPU:
			std::cout << "Bee boop bad CPU!" << std::endl;
			break;
		case RTC_CANCELLED:
			std::cout << "DENIED!" << std::endl;
			break;
		}
	}


	void import_yocto_meshes(yobj::scene * scn, bree_container & breec) {
		std::map<std::string, unsigned int> msh_scn;
		unsigned int id;
		for (size_t mesh_no = 0; mesh_no < scn->meshes.size(); mesh_no++) {
			auto msh = scn->meshes[mesh_no];
			auto shape = msh->shapes[0];
			auto positions = shape->pos;
			auto radii = shape->radius;
			size_t vertex_no = positions.size();
			auto &rtcscn = breec.bree_scenes.emplace_back(rtcDeviceNewScene(breec.bree_device, RTC_SCENE_STATIC, RTC_INTERSECT1));

			// Determine the type of shape
			size_t points_size = shape->points.size();
			size_t lines_size = shape->lines.size();
			size_t triangles_size = shape->triangles.size();

			// And at 1:37 of 04/11 the programmer typed, writing: "Points shalt be represented
			// as _very_ short lines. And each point will actually be two points together, no more
			// no less. Two shall be the number of points per points and the number of points shall be two
			// A triangle shalt thou not you use, neither shalt thou use a point, , excepting that thou 
			// then proceed to a line. A quad is right out! Once the number two, being the second number
			// be reached, then assign thou thy points to a new line in the buffer. Amen.
			if (points_size != 0) {
				id = rtcNewLineSegments2(rtcscn, RTC_GEOMETRY_STATIC, points_size, vertex_no, 1, mesh_no);
				auto points = (point*) rtcMapBuffer(rtcscn, id, RTC_INDEX_BUFFER);
				//auto err_no = rtcDeviceGetError(breec.bree_device);
				//print_error("Wassup my points", err_no);

				for (size_t p = 0; p < points_size; p++){
					points[p].v0 = shape->points[p];
				}
			}				
			else if (lines_size!= 0) {
				id = rtcNewLineSegments2(rtcscn, RTC_GEOMETRY_STATIC, lines_size, vertex_no, 0, mesh_no);
				auto lines = (line*) rtcMapBuffer(rtcscn, id, RTC_INDEX_BUFFER);
				//auto err_no = rtcDeviceGetError(breec.bree_device);
				//print_error("Wassup my lines", err_no);

				for (size_t l = 0; l < lines_size; l++) {
					lines[l].v0 = shape->lines[l].x;
					//lines[l].v1 = shape->lines[l].y;
				}
			}
			else if (triangles_size != 0) {
				id = rtcNewTriangleMesh2(rtcscn, RTC_GEOMETRY_STATIC, triangles_size, vertex_no, 1, mesh_no);
				auto triangles = (triangle*) rtcMapBuffer(rtcscn, id, RTC_INDEX_BUFFER);
				//auto err_no = rtcDeviceGetError(breec.bree_device);
				//print_error("Wassup my triangles", err_no);

				for (size_t t = 0; t < triangles_size; t++) {
					triangles[t].v0 = shape->triangles[t].x;
					triangles[t].v1 = shape->triangles[t].y;
					triangles[t].v2 = shape->triangles[t].z;
				}
			}

			auto vertices = (vertex*)rtcMapBuffer(rtcscn, id, RTC_VERTEX_BUFFER);
			for (size_t v = 0; v < vertex_no; v++) {
				auto &ver = vertices[v];
				auto &cver = positions[v];
				ver.x = cver.x;
				ver.y = cver.y;
				ver.z = cver.z;

				if (triangles_size == 0) {
					auto cradius = radii[v];
					ver.a = cradius;
				}
				else {
					ver.a = FLT_EPSILON;
				}
			}

			rtcUnmapBuffer(rtcscn, id, RTC_INDEX_BUFFER);
			rtcUnmapBuffer(rtcscn, id, RTC_VERTEX_BUFFER);

			// Add the shape id to the vector indexed by the mesh
			rtcCommit(rtcscn);
			msh_scn[msh->name] = id;
		}

		for (size_t instance_no = 0; instance_no < scn->instances.size(); instance_no++) {
			auto instance = scn->instances[instance_no];

			auto &mat = breec.instance_transformations.emplace_back(flat4matrix{});
			auto xform = instance->xform();
			copy_matrix(mat, xform);

			auto &scene = breec.bree_scenes[msh_scn[instance->msh->name]];
			id = rtcNewInstance3(breec.main_scene, scene, 1, instance_no);
			rtcSetTransform2(breec.main_scene, id, RTC_MATRIX_COLUMN_MAJOR, (float *)(mat.matrix), 1);
			rtcUpdate(breec.main_scene, id);
		}

		rtcCommit(breec.main_scene);
		
	}

	RTCRay build_ray(ym::vec3f o, ym::vec3f dir, float tnear = 1e-6f, float tfar = INFINITY) {
		RTCRay ray;
		ray.org[0] = o.x;  ray.org[1] = o.y; ray.org[2] = o.z;
		ray.dir[0] = dir.x; ray.dir[1] = dir.y; ray.dir[2] = dir.z;
		ray.tnear = tnear;
		ray.tfar = tfar;
		ray.geomID = RTC_INVALID_GEOMETRY_ID;
		ray.instID = RTC_INVALID_GEOMETRY_ID;
		ray.primID = RTC_INVALID_GEOMETRY_ID;
		ray.mask = -1;
		ray.time = 0;

		return ray;
	}


	// Use this one for 3-dimensional vectors
	ym::vec3f barycentric_to_vec(float u, float v, std::vector<ym::vec3f> vecs, int vec_no) {
		ym::vec3f vec = { 0, 0, 0 };
		ym::vec3f bar_coord = { 1 - u - v, u, v };
		for (int i = 0; i < vec_no; i++) {
			vec += bar_coord[i] * vecs[i];
		}
		return vec;
	}

	// Use this one 2 dimensional vectors ( for textures )
	ym::vec2f barycentric_to_vec(float u, float v, std::vector<ym::vec2f> vecs, int vec_no) {
		ym::vec2f vec = { 0, 0 };
		ym::vec3f bar_coord = { 1 - u - v, u, v };
		for (int i = 0; i < vec_no; i++) {
			vec += bar_coord[i] * vecs[i];
		}
		return vec;
	}

	// ---------------------------------------------------------------------------------------------------- //
	// Should I manually copy and paste from the slides, possibly introducing errors or just use a pre-made
	// function that I know works ? To be fair though, a different interpolation algorithm is provided

	// It's got asserts, it's what functions crave.
	inline ym::vec4f lookup_texture(const yobj::texture* txt, const ym::vec2i& ij, bool srgb ) {
		if (txt->ldr) {
			auto v = txt->ldr[{ij.x, ij.y}];
			return (srgb) ? ym::srgb_to_linear(v) : ym::byte_to_float(v);
		}
		else if (txt->hdr) {
			return txt->hdr[{ij.x, ij.y}];
		}
		else {
			assert(false);
			return {};
		}
	}

	ym::vec4f eval_texture(const yobj::texture* txt, const ym::vec2f& texcoord, bool srgb ) {
		// And here's where I'd put my texture...
		if (!txt) {
			// ... IF I HAD ONE!
			return { 1.f, 1.f, 1.f, 1.f };
		}

		auto wh = ym::vec2i{ txt->width(), txt->height() };

		// get coordinates normalized for tiling
		auto st = ym::vec2f{
			std::fmod(texcoord.x, 1.0f) * wh.x, std::fmod(texcoord.y, 1.0f) * wh.y };

		// wtf, texture coordinates can have negative values ???
		if (st.x < 0) st.x += wh.x;
		if (st.y < 0) st.y += wh.y;

		// get image coordinates and residuals
		auto ij = ym::clamp(ym::vec2i{ (int)st.x, (int)st.y }, { 0, 0 }, wh);
		auto uv = st - ym::vec2f{ (float)ij.x, (float)ij.y };

		// When you're programming in C++ but tuples are life
		// get interpolation weights and indices
		ym::vec2i idx[4] = { ij,{ ij.x, (ij.y + 1) % wh.y },
		{ (ij.x + 1) % wh.x, ij.y },{ (ij.x + 1) % wh.x, (ij.y + 1) % wh.y } };
		auto w = ym::vec4f{ (1 - uv.x) * (1 - uv.y), (1 - uv.x) * uv.y,
			uv.x * (1 - uv.y), uv.x * uv.y };

		// handle interpolation
		return (lookup_texture(txt, idx[0], srgb) * w.x +
				lookup_texture(txt, idx[1], srgb) * w.y +
				lookup_texture(txt, idx[2], srgb) * w.z +
				lookup_texture(txt, idx[3], srgb) * w.w);
	}

	// ---------------------------------------------------------------------------------------------------- //

	// Compute the opaqueness at the intersection point
	float opaqueness_at_intersection(const yobj::scene* scn, RTCRay & intersection) {
		float op;
		auto instance = scn->instances[intersection.instID];
		auto mesh = instance->msh;
		auto shape = mesh->shapes[0];
		auto mat = shape->mat;

		op = mat->opacity;

		if (shape->texcoord.size() > 0 && mat->op_txt) {
			ym::vec2f texture_coords;
			if (shape->points.size() != 0) {
				auto p = shape->points[intersection.primID];
				texture_coords = shape->texcoord[p];
			}
			else if (shape->lines.size() != 0) {
				auto l = shape->lines[intersection.primID];
				std::vector<ym::vec2f> textures = { shape->texcoord[l[0]], shape->texcoord[l[1]] };
				texture_coords = barycentric_to_vec(intersection.u, intersection.v, textures, 2);
			}
			else if (shape->triangles.size() != 0) {
				auto t = shape->triangles[intersection.primID];
				std::vector<ym::vec2f> textures = { shape->texcoord[t[0]], shape->texcoord[t[1]], shape->texcoord[t[2]] };
				texture_coords = barycentric_to_vec(intersection.u, intersection.v, textures, 3);
			}

			op *= eval_texture(mat->op_txt, texture_coords, false)[0];
		}

		return op;
	}

	point_info get_point_info(const yobj::scene* scn, RTCRay & intersection, bool is_textured ) {
		point_info p_i;
		auto shape = scn->instances[intersection.instID]->msh->shapes[0];

		// Determine the type of shape
		if (shape->points.size() != 0) {
			auto p = shape->points[intersection.primID];
			p_i.point = shape->pos[p];
			p_i.normal = shape->norm[p];
			if (is_textured) {
				p_i.texture_coords = shape->texcoord[p];
			}
		}
		else if (shape->lines.size() != 0) {
			auto l = shape->lines[intersection.primID];
			std::vector<ym::vec3f> vectors = { shape->pos[l[0]], shape->pos[l[1]] };
			std::vector<ym::vec3f> normals = { shape->norm[l[0]], shape->norm[l[1]] };

			// Goodbye simmetry hello efficiency
			if (is_textured) {
				std::vector<ym::vec2f> textures = { shape->texcoord[l[0]], shape->texcoord[l[1]] };
				p_i.texture_coords = barycentric_to_vec(intersection.u, intersection.v, textures, 2);
			}

			p_i.point = barycentric_to_vec(intersection.u, intersection.v, vectors, 2);
			p_i.normal = ym::normalize(barycentric_to_vec(intersection.u, intersection.v, normals, 2));
		}
		else if (shape->triangles.size() != 0) {
			auto t = shape->triangles[intersection.primID];
			std::vector<ym::vec3f> vectors = { shape->pos[t[0]], shape->pos[t[1]], shape->pos[t[2]] };
			std::vector<ym::vec3f> normals = { shape->norm[t[0]], shape->norm[t[1]], shape->norm[t[2]] };

			// Goodbye simmetry hello efficiency
			if (is_textured) {
				std::vector<ym::vec2f> textures = { shape->texcoord[t[0]], shape->texcoord[t[1]], shape->texcoord[t[2]] };
				p_i.texture_coords = barycentric_to_vec(intersection.u, intersection.v, textures, 3);
			}

			p_i.point = barycentric_to_vec(intersection.u, intersection.v, vectors, 3);
			p_i.normal = ym::normalize(barycentric_to_vec(intersection.u, intersection.v, normals, 3));
		}

		return p_i;
	};

	ym::vec4f shade_light(const yobj::scene* scn, bree_container & breec, const yobj::instance* light, const int light_point, RTCRay & ray, bool coloured_light ) {
		ym::vec3f light_colour;

		auto light_shape = light->msh->shapes[0];
		auto light_mat = light_shape->mat;
		auto light_transform = light->xform();

		auto light_pos = light_shape->pos[light_point];
		// Bring back to world coord
		light_pos = ym::transform_point(light_transform, light_pos);

		auto light_direction = ym::normalize(light_pos - ym::vec3f{ray.org[0], ray.org[1], ray.org[2]} );
		auto light_distance = ym::length(light_pos - ym::vec3f{ ray.org[0], ray.org[1], ray.org[2] });

		auto light_intersection = build_ray(ym::vec3f{ ray.org[0], ray.org[1], ray.org[2] }, light_direction, 1e-6f, light_distance);
		rtcIntersect(breec.main_scene, light_intersection);


		// Hello there!
		// General Bodgeroni
		const yobj::instance *x_int;
		if (light_intersection.geomID != RTC_INVALID_GEOMETRY_ID) {
			x_int = light;
		}
		if (light_intersection.instID < scn->instances.size()) {
			x_int = scn->instances[light_intersection.instID];
		}

		// Dafuq did we hit?
		if (coloured_light) {
			if (light_intersection.geomID != RTC_INVALID_GEOMETRY_ID && !(x_int == light)) {
				auto light_hit_instance = scn->instances[light_intersection.instID];
				float light_op;
				if ((light_op = opaqueness_at_intersection(scn, light_intersection)) < 1) {
					auto mat = light_hit_instance->msh->shapes[0]->mat;
					auto kd = mat->kd;
					bool is_textured = (mat->kd_txt) ? true : false;

					auto light_hit_point_i = get_point_info(scn, light_intersection, is_textured);
					auto light_hit_point = light_hit_point_i.point;

					if (is_textured) {
						auto kd_4 = eval_texture(mat->kd_txt, light_hit_point_i.texture_coords, true);
						kd *= { kd_4[0], kd_4[1], kd_4[2] };
					}

					light_hit_point = ym::transform_point(light_hit_instance->xform(), light_hit_point);
					auto light_hit_colour = shade_light(scn, breec, light, light_point, build_ray(light_hit_point, light_direction, 1e-6f));
					if (light_hit_colour[3] < 1.f) {
						return (1.f - light_op) * ym::vec4f{ kd[0] * light_hit_colour[0], kd[1] * light_hit_colour[1], kd[2] * light_hit_colour[2], 1.f };
					}
				}
				// President Shader, an opaque body is blocking our lights in Brazil
				// SHUT. DOWN. EVERYTHING!
				return { 0.f, 0.f, 0.f, 1.f };
			}
		}
		else {
			if (light_intersection.geomID != RTC_INVALID_GEOMETRY_ID && !(x_int == light)) {
				return { 0.f, 0.f, 0.f, 1.f };
			}
		}

		// If we intersect a light we can finally return a light intensity!
		return { light_mat->ke[0], light_mat->ke[1], light_mat->ke[2], 0.f };
	}

	ym::vec4f shade(const yobj::scene * scn, bree_container & breec, const std::vector<yobj::instance*>& lights, const ym::vec3f & amb, RTCRay & ray, int ray_depth, bool coloured_light) {
		rtcIntersect(breec.main_scene, ray);
		if (ray.geomID != RTC_INVALID_GEOMETRY_ID) {
			if (ray.instID != RTC_INVALID_GEOMETRY_ID){
				// Initialize colour with ambient light
				ym::vec3f colour = { 0.f, 0.f, 0.f };

				auto instance = scn->instances[ray.instID];
				auto mesh = instance->msh;
				auto shape = mesh->shapes[0];
				auto mat = shape->mat;

				// Coefficients
				// Specular coefficient
				ym::vec3f ks;
				// Diffuse coefficient
				ym::vec3f kd;
				// Reflective coefficient
				ym::vec3f kr;
				// Roughness coefficient
				float rs;
				// Opacity coefficient
				float op;

				ym::vec3f point;
				ym::vec3f normal;

				bool is_textured = false;
				ym::vec2f texture_coords;

				if (shape->texcoord.size() > 0) {
					is_textured = true;
				}

				auto p_i = get_point_info(scn, ray, is_textured);
				point = p_i.point;
				normal = p_i.normal;
				texture_coords = p_i.texture_coords;


				// Set the coefficients
				ks = mat->ks;
				kd = mat->kd;
				rs = mat->rs;
				kr = mat->kr;
				op = mat->opacity;

				// Compute the texture_coords value
				if (is_textured) {
					auto op_4 = eval_texture(mat->op_txt, texture_coords, false);
					op *= op_4[0];

					auto ks_4 = eval_texture(mat->ks_txt, texture_coords, true);
					ks *= ym::vec3f{ ks_4[0], ks_4[1], ks_4[2] };

					auto kd_4 = eval_texture(mat->kd_txt, texture_coords, true);
					kd *= ym::vec3f{ kd_4[0], kd_4[1], kd_4[2] };

					auto kr_4 = eval_texture(mat->kr_txt, texture_coords, false);
					kr *= { kr_4[0], kr_4[1], kr_4[2]};

					auto rs_4 = eval_texture(mat->rs_txt, texture_coords, false);
					rs *= rs_4[0];
				}

				bool is_reflective = (kr[0] > 0.f || kr[1] > 0.f || kr[2] > 0.f);
				bool is_specular = (ks[0] > 0.f || ks[1] > 0.f || ks[2] > 0.f);

				// Bring back to world coordinates
				point = ym::transform_point(instance->xform(), point);
				normal = ym::transform_direction(instance->xform(), normal);

				// Compute the vector from point to eye
				auto eye_vector = ym::vec3f{ -ray.dir[0], -ray.dir[1], -ray.dir[2] };

				float bp_exponent;
				if (is_specular && !is_reflective) {
					bp_exponent = (rs > 0.f) ? 2.f / (rs * rs) - 2.f : 1e6f;
				}

				for (auto light : lights) {
					auto light_shape = light->msh->shapes[0];
					auto light_mat = light_shape->mat;
					auto light_transform = light->xform();

					for (auto light_point : light_shape->points) {
						auto light_pos = light_shape->pos[light_point];
						// Bring back to world coord
						light_pos = ym::transform_point(light_transform, light_pos);

						auto light_direction = ym::normalize(light_pos - point);
						auto light_distance = ym::length(light_pos - point);

						auto shaded_light = shade_light(scn, breec, light, light_point, build_ray(point, light_direction, 1e-6f, light_distance), coloured_light);

						if (shaded_light[3] != 1.f) {
							auto light_intensity = ym::vec3f{ shaded_light[0], shaded_light[1], shaded_light[2] } / (light_distance * light_distance);


							if (shape->lines.size() == 0) {
								// Mary had a little Lambert, little Lambert, little Lambert...
								colour += kd * light_intensity * ym::max(0.0f, ym::dot(normal, light_direction));

								// ---------------------------------------------------------------------------------------------------- //

								// I know when the hotline Blinn, it can only mean one Phong
								if (is_specular && !is_reflective) {
									auto eye_plus_light = eye_vector + light_direction;
									auto bisector = ym::normalize(eye_plus_light);

									colour += ks * light_intensity * ym::pow(ym::max(0.0f, ym::dot(normal, bisector)), bp_exponent);
								}
							}
							else {
								// Déjà vu, much?
								auto eye_plus_light = eye_vector + light_direction;
								auto bisector = ym::normalize(eye_plus_light);

								colour += kd * light_intensity * ym::sqrt(1 - ym::abs(ym::dot(normal, light_direction))) + ks * light_intensity *  ym::pow(ym::sqrt(1 - ym::abs(ym::dot(normal, bisector))), bp_exponent);
							}

						}
					}
				}

				// Mirror mirror on the scene...
				if ((ray_depth < MAX_BOUNCES) && is_reflective) {
					auto mirror_ray = ym::normalize(2.f * ym::dot(normal, eye_vector)*normal - eye_vector);
					auto mirror_intersection = build_ray(point, mirror_ray);
					rtcIntersect(breec.main_scene, mirror_intersection);

					if (mirror_intersection.geomID != RTC_INVALID_GEOMETRY_ID) {
						auto reflected_colour = shade(scn, breec, lights, amb, mirror_intersection, ray_depth + 1, coloured_light);
						ym::vec3f ref_col = { reflected_colour[0], reflected_colour[1], reflected_colour[2] };
						colour += kr * ref_col;
					}
				}

				// Since ambient light comes from all directions we can assume there is at least 1 ray of ambient light perfectly
				// specular to our surface. We can thus sum it to the colour
				colour += kd * amb;
				ym::vec4f full_colour = { colour[0], colour[1], colour[2], 1.f };

				// You-u-uh see right through me
				if (op < 1) {
					full_colour = op*full_colour + (1 - op)*shade(scn, breec, lights, amb, build_ray(point, {ray.dir[0], ray.dir[1], ray.dir[2]}), ray_depth, coloured_light);
				}

				return full_colour;
			}

			// Eh, maybe?
			return {};
		}

		return ym::vec4f();
	}

	RTCRay eval_camera(const yobj::camera* cam, const ym::vec2f& uv) {
		auto height = 2 * ym::tan(cam->yfov / 2);
		auto width = height * cam->aspect;

		ym::frame3f frame = ym::to_frame(cam->xform());
		ym::vec4f q = cam->xform() * ym::vec4f{ ((uv[0] - 0.5f) * width), ((uv[1] - 0.5f) * height), -1, 1 };
		ym::vec3f dir = ym::normalize(ym::vec3f{ q[0], q[1], q[2] } - frame.o);

		RTCRay ray;
		ray.org[0] = frame.o.x;  ray.org[1] = frame.o.y; ray.org[2] = frame.o.z;
		ray.dir[0] = dir.x; ray.dir[1] = dir.y; ray.dir[2] = dir.z;
		ray.tnear = 1e-6f;
		ray.tfar = INFINITY;
		ray.geomID = RTC_INVALID_GEOMETRY_ID;
		ray.primID = RTC_INVALID_GEOMETRY_ID;
		ray.mask = -1;
		ray.time = 0;

		return ray;
	}

	ym::image4f bree_raytrace(bree_container & breec, yobj::scene * scn, const ym::vec3f & amb, int resolution, int samples, int cam_no, bool coloured_light) {
		yu::logging::log_info("Started embree parallel tracing");

		auto camera = scn->cameras[cam_no];
		auto height = resolution;
		auto width = ym::floor(height * camera->aspect);


		//std::cout << "Img height: " << height << " image width: " << width << std::endl;
		ym::image4f pixels = ym::image4f(width, height, { 0, 0, 0, 0 });

		std::vector<yobj::instance*> lights;
		// Find the lights in the scene
		for each (auto instance in scn->instances) {
			if (instance->msh->shapes[0]->points.size() > 0) {
				auto ke = instance->msh->shapes[0]->mat->ke;
				if ((ke[0] + ke[1] + ke[2]) > 0) {
					lights.push_back(instance);
				}
			}
		}

		// Compute the image
	#pragma omp parallel for num_threads(8) 
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				for (int si = 0; si < samples; si++) {
					for (int sj = 0; sj < samples; sj++) {
						auto u = (j + (sj + 0.5f) / samples) / width;
						auto v = (i + (si + 0.5f) / samples) / height;

						auto ray = eval_camera(camera, { u, 1 - v });
						pixels[{j, i}] += shade(scn, breec, lights, amb, ray, 0, coloured_light);
					}
				}

				// Add the central pixel just to be extra sure
				auto u = (j + 0.5f) / width;
				auto v = (i + 0.5f) / height;

				auto ray = eval_camera(camera, { u, 1 - v });
				pixels[{j, i}] += shade(scn, breec, lights, amb, ray, 0, coloured_light);

				pixels[{j, i}] /= ((float)(samples * samples) + 1.f);
			}
		}

		// Construct and return an img;
		return pixels;
	}	
}
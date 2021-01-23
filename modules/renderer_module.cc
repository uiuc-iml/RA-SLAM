#include "modules/renderer_module.h"

ImageRenderer::ImageRenderer(const std::string &name,
                const std::shared_ptr<SLAMSystem> &slam,
                const std::shared_ptr<TSDFSystem> &tsdf,
                const std::string &config_file_path)
     : RendererBase(name),
       slam_(slam),
       tsdf_(tsdf),
       map_publisher_(slam->get_map_publisher()),
       config_(YAML::LoadFile(config_file_path)),
       virtual_cam_(get_intrinsics_from_file(config_file_path), 360, 640) {
    ImGuiIO &io = ImGui::GetIO();
    io.FontGlobalScale = 2;
}

void ImageRenderer::DispatchInput() {
    ImGuiIO &io = ImGui::GetIO();
    if (io.MouseWheel != 0) {
        follow_cam_ = false;
        const Vector3<float> move_cam(0, 0, io.MouseWheel * .1);
        const SO3<float> virtual_cam_R_world = virtual_cam_P_world_.GetR();
        const Vector3<float> virtual_cam_T_world = virtual_cam_P_world_.GetT();
        virtual_cam_P_world_ = SE3<float>(virtual_cam_R_world, virtual_cam_T_world - move_cam);
    }
    if (!io.WantCaptureMouse && ImGui::IsMouseDragging(0) && tsdf_normal_.width) {
        follow_cam_ = false;
        const ImVec2 delta = ImGui::GetMouseDragDelta(0);
        const Vector2<float> delta_img(delta.x / io.DisplaySize.x * tsdf_normal_.width,
                                     delta.y / io.DisplaySize.y * tsdf_normal_.height);
        const Vector2<float> pos_new_img(io.MousePos.x / io.DisplaySize.x * tsdf_normal_.width,
                                       io.MousePos.y / io.DisplaySize.y * tsdf_normal_.height);
        const Vector2<float> pos_old_img = pos_new_img - delta_img;
        const Vector3<float> pos_new_cam = virtual_cam_.intrinsics_inv * Vector3<float>(pos_new_img);
        const Vector3<float> pos_old_cam = virtual_cam_.intrinsics_inv * Vector3<float>(pos_old_img);
        const Vector3<float> pos_new_norm_cam = pos_new_cam / sqrt(pos_new_cam.dot(pos_new_cam));
        const Vector3<float> pos_old_norm_cam = pos_old_cam / sqrt(pos_old_cam.dot(pos_old_cam));
        const Vector3<float> rot_axis_cross_cam = pos_new_norm_cam.cross(pos_old_norm_cam);
        const float theta = acos(pos_new_norm_cam.dot(pos_old_norm_cam));
        const Vector3<float> w = rot_axis_cross_cam / sin(theta) * theta;
        const Matrix3<float> w_x(0, -w.z, w.y, w.z, 0, -w.x, -w.y, w.x, 0);
        const Matrix3<float> R = Matrix3<float>::Identity() +
                               (float)sin(theta) / theta * w_x +
                               (float)(1 - cos(theta)) / (theta * theta) * w_x * w_x;
        const SE3<float> pose_cam1_P_cam2(R, Vector3<float>(0));
        virtual_cam_P_world_ = pose_cam1_P_cam2.Inverse() * virtual_cam_P_world_old_;
    }
    else if (!io.WantCaptureMouse && ImGui::IsMouseDragging(2)) {
        follow_cam_ = false;
        const ImVec2 delta = ImGui::GetMouseDragDelta(2);
        const Vector3<float> translation(delta.x, delta.y, 0);
        const Vector3<float> T = virtual_cam_P_world_old_.GetT();
        const Matrix3<float> R = virtual_cam_P_world_old_.GetR();
        virtual_cam_P_world_ = SE3<float>(R, T + translation * .01);
    }
    else {
        virtual_cam_P_world_old_ = virtual_cam_P_world_;
    }
}

void ImageRenderer::Render() {
    int display_w, display_h;
    glfwGetFramebufferSize(window_, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    // GUI
    ImGui::Begin("Menu");
    if (ImGui::Button("Follow Camera")) { follow_cam_ = true; }
    // compute
    const auto m = map_publisher_->get_current_cam_pose();
    cam_P_world_ = SE3<float>(
        m(0, 0), m(0, 1), m(0, 2), m(0, 3),
        m(1, 0), m(1, 1), m(1, 2), m(1, 3),
        m(2, 0), m(2, 1), m(2, 2), m(2, 3),
        m(3, 0), m(3, 1), m(3, 2), m(3, 3)
    );
    if (!tsdf_normal_.height || !tsdf_normal_.width) {
        tsdf_normal_.BindImage(virtual_cam_.img_h, virtual_cam_.img_w, nullptr);
    }
    if (follow_cam_) {
        static float step = 0;
        ImGui::SliderFloat("behind actual camera", &step, 0.0f, 3.0f);
        virtual_cam_P_world_ = SE3<float>(cam_P_world_.GetR(),
                                        cam_P_world_.GetT() + Vector3<float>(0, 0, step));
    }
    // render
    const auto st = get_timestamp<std::chrono::milliseconds>();
    tsdf_->Render(virtual_cam_, virtual_cam_P_world_, &tsdf_normal_);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    const auto end = get_timestamp<std::chrono::milliseconds>();
    ImGui::Text("Rendering takes %lu ms", end - st);
    tsdf_normal_.Draw();
    ImGui::End();
}

void ImageRenderer::RenderExit() {
    slam_->request_terminate();
}
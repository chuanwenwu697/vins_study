#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {

        it.used_num = it.feature_per_frame.size();

        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}

/**
 * @brief 把特征点放入feature的list容器中，计算每一个点跟踪次数和它在次新帧和次次新帧间的视差，返回是否是关键帧
 * 
 * @param frame_count 窗口内帧的个数
 * @param image 某帧所有特征点的[camera_id,[x,y,z,u,v,vx,vy]]s构成的map,索引为feature_id
 * @param td IMU和cam同步时间差
 * @return true 次新帧是关键帧;false：非关键帧
 * @return false 
 */

bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    //ROS_DEBUG("input feature: %d", (int)image.size());
    //ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;    //总视差
    int parallax_num = 0;       //次新帧与次次新帧中的共同特征的数量
    last_track_num = 0;         //被跟踪点的个数，非新特征点的个数
    //遍历图像image中所有的特征点，和已经记录了特征点的容器feature中进行比较
    for (auto &id_pts : image)
    {
        
        //特征点在观测到其的第一幅图像中的xyz_uv_velocity 
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

        
        //特征点的编号
        int feature_id = id_pts.first;
        //find_if函数，寻找列表中第一个使判别式为true的元素       
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });
        //如果没有则新建一个，并添加这图像帧
        if (it == feature.end())
        {
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
        }//有的话把图像帧添加进去
        else if (it->feature_id == feature_id)
        {
            /**
             * 如果找到了相同ID特征点，就在其FeaturePerFrame内增加此特征点在此帧的位置以及其他信息，
             * it的feature_per_frame容器中存放的是该feature能够被哪些帧看到，存放的是在这些帧中该特征点的信息
             * 所以，feature_per_frame.size的大小就表示有多少个帧可以看到该特征点
             * */
            it->feature_per_frame.push_back(f_per_fra);
            //last_track_num，表示此帧有多少个和其他帧中相同的特征点能够被追踪到
            last_track_num++;
        }
    }
    //如果窗口内只有一帧或者跟踪到的特征点小于20个，为关键帧
    if (frame_count < 2 || last_track_num < 20)
        return true;

    //计算每个特征在次新帧和次次新帧中的视差
    //遍历每一个feature
    for (auto &it_per_id : feature)
    {
        //计算能被当前帧和其前两帧共同看到的特征点视差
        //it_per_id.feature_per_frame.size()表示该特征点能够被多少帧共视
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            //计算特征点it_per_id在倒数第二帧和倒数第三帧之间的视差，并求所有视差的累加和
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            //所有具有视差的特征个数
            parallax_num++;
        }
    }
    //视差等于零，说明没有共同特征
    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        //ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        //ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        //视差总和除以参与计算视差的特征点个数，表示每个特征点的平均视差值
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    //ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        assert(it.feature_per_frame.size() != 0);
        assert(it.start_frame >= 0);
        assert(it.used_num >= 0);

        //ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            //ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        assert(it.used_num == sum);
    }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;//当前帧第一次观测到特征点的帧数
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

VectorXd FeatureManager::getDepthVector()
{   //求逆深度VectorXd
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{   // 遍历滑窗每个特征点,求出该特征点在它start_frame坐标下的深度
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size(); // 每个id的特征点被多少帧图像观测到了
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            // 如果该特征点被两帧及两帧以上的图像观测到，
            // 且观测到该特征点的第一帧图像应该早于或等于滑动窗口第9最新关键帧
            // 也就是说，至少是第9最新关键帧和第8最新关键帧观测到了该特征点  （第2最新帧似乎是紧耦合优化的最新帧）
            continue;

        if (it_per_id.estimated_depth > 0)//该id的特征点深度值大于0，该值在初始化时为-1，如果大于0，说明该点被三角化过
            continue;
        // imu_i：观测到该特征点的第一帧图像在滑动窗口中的帧号
        // imu_j：观测到该特征点的最后一帧图像在滑动窗口中的帧号
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        assert(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        // start_frame处的位姿 Twc w世界坐标
        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();//单位旋转矩阵
        P0.rightCols<1>() = Eigen::Vector3d::Zero();//0平移向量
        // 遍历观测到该id特征点的每一图像帧
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;// 观测到该特征点的最后一帧图像在滑动窗口中的帧号
            
            // 下面要求出  Tc1c0 
            // 求第二个相机的位姿   Twc    相对于l相机坐标系
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];// Pwi+ Rwi*tic   = Pwc
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0]; // rwi*ric   = rwc
            //  C1->C0    Tc0c1  
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);// Rc0l*(t01)  =  在C0系下的t01  
            Eigen::Matrix3d R = R0.transpose() * R1;// Rc0l*Rlc1  = Rc0c1 
            Eigen::Matrix<double, 3, 4> P;
            // C0->C1      Tc1c0  
            P.leftCols<3>() = R.transpose(); // Rc1c0 
            P.rightCols<1>() = -R.transpose() * t;// - Rc1c0 * t01  = T10
            Eigen::Vector3d f = it_per_frame.point.normalized(); // C1的归一化坐标 
            // 这个时候求的该特征点在C0坐标系下的3D位置
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        assert(svd_idx == svd_A.rows());
        // 求解 SVD   UWVt    最右奇异向量
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];// Z轴 
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];
        // 注意  estimated_depth表示的是在特征点在其start_frame下的深度
        it_per_id.estimated_depth = svd_method; // 得到了该特征点的深度
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id.estimated_depth < 0.1)//如果估计出来的深度小于0.1（单位是m），则把它替换为一个设定的值
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }

    }
}

void FeatureManager::removeOutlier()
{
    // ROS_BREAK();
    return;
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)//特征点it的初始帧不是第0帧，则减减(所有点的初始帧号减１)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);//删除次新帧所对应的特征点
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}
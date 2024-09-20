
#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyActivationListener.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <Jolt/Physics/Collision/PhysicsMaterial.h>
#include <Jolt/Physics/Collision/CollisionDispatch.h>

#include <stdarg.h>
#include <ctype.h>
#include <stdio.h>
#include <iostream>
#include <thread>

using namespace JPH;
using namespace std;

namespace RL {
  #include <raylib.h>
}


// Callback for traces, connect this to your own trace function if you have one
static void TraceImpl(const char *inFMT, ...)
{
    // Format the message
    va_list list;
    va_start(list, inFMT);
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), inFMT, list);
    va_end(list);

    // Print to the TTY
    cout << buffer << endl;
}

#ifdef JPH_ENABLE_ASSERTS

// Callback for asserts, connect this to your own assert handler if you have one
static bool AssertFailedImpl(const char *inExpression, const char *inMessage, const char *inFile, uint inLine)
{
    // Print to the TTY
    cout << inFile << ":" << inLine << ": (" << inExpression << ") " << (inMessage != nullptr ? inMessage : "") << endl;

    // Breakpoint
    return true;
};

#endif // JPH_ENABLE_ASSERTS

// Layer that objects can be in, determines which other objects it can collide with
// Typically you at least want to have 1 layer for moving bodies and 1 layer for static bodies, but you can have more
// layers if you want. E.g. you could have a layer for high detail collision (which is not used by the physics simulation
// but only if you do collision testing).
namespace Layers
{
    static constexpr ObjectLayer NON_MOVING = 0;
    static constexpr ObjectLayer MOVING = 1;
    static constexpr ObjectLayer NUM_LAYERS = 2;
};

/// Class that determines if two object layers can collide
class ObjectLayerPairFilterImpl : public ObjectLayerPairFilter
{
public:
    virtual bool ShouldCollide(ObjectLayer inObject1, ObjectLayer inObject2) const override
    {
        switch (inObject1)
        {
        case Layers::NON_MOVING:
            return inObject2 == Layers::MOVING; // Non moving only collides with moving
        case Layers::MOVING:
            return true; // Moving collides with everything
        default:
            JPH_ASSERT(false);
            return false;
        }
    }
};

// Each broadphase layer results in a separate bounding volume tree in the broad phase. You at least want to have
// a layer for non-moving and moving objects to avoid having to update a tree full of static objects every frame.
// You can have a 1-on-1 mapping between object layers and broadphase layers (like in this case) but if you have
// many object layers you'll be creating many broad phase trees, which is not efficient. If you want to fine tune
// your broadphase layers define JPH_TRACK_BROADPHASE_STATS and look at the stats reported on the TTY.
namespace BroadPhaseLayers
{
    static constexpr BroadPhaseLayer NON_MOVING(0);
    static constexpr BroadPhaseLayer MOVING(1);
    static constexpr uint NUM_LAYERS(2);
};

// BroadPhaseLayerInterface implementation
// This defines a mapping between object and broadphase layers.
class BPLayerInterfaceImpl final : public BroadPhaseLayerInterface
{
public:
    BPLayerInterfaceImpl()
    {
        // Create a mapping table from object to broad phase layer
        mObjectToBroadPhase[Layers::NON_MOVING] = BroadPhaseLayers::NON_MOVING;
        mObjectToBroadPhase[Layers::MOVING] = BroadPhaseLayers::MOVING;
    }

    virtual uint GetNumBroadPhaseLayers() const override
    {
        return BroadPhaseLayers::NUM_LAYERS;
    }

    virtual BroadPhaseLayer GetBroadPhaseLayer(ObjectLayer inLayer) const override
    {
        JPH_ASSERT(inLayer < Layers::NUM_LAYERS);
        return mObjectToBroadPhase[inLayer];
    }

#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    virtual const char *GetBroadPhaseLayerName(BroadPhaseLayer inLayer) const override
    {
        switch ((BroadPhaseLayer::Type)inLayer)
        {
        case (BroadPhaseLayer::Type)BroadPhaseLayers::NON_MOVING:
            return "NON_MOVING";
        case (BroadPhaseLayer::Type)BroadPhaseLayers::MOVING:
            return "MOVING";
        default:
            JPH_ASSERT(false);
            return "INVALID";
        }
    }
#endif // JPH_EXTERNAL_PROFILE || JPH_PROFILE_ENABLED

private:
    BroadPhaseLayer mObjectToBroadPhase[Layers::NUM_LAYERS];
};

/// Class that determines if an object layer can collide with a broadphase layer
class ObjectVsBroadPhaseLayerFilterImpl : public ObjectVsBroadPhaseLayerFilter
{
public:
    virtual bool ShouldCollide(ObjectLayer inLayer1, BroadPhaseLayer inLayer2) const override
    {
        switch (inLayer1)
        {
        case Layers::NON_MOVING:
            return inLayer2 == BroadPhaseLayers::MOVING;
        case Layers::MOVING:
            return true;
        default:
            JPH_ASSERT(false);
            return false;
        }
    }
};

static JobSystemThreadPool *g_job_system = NULL;
static TempAllocatorImpl *g_temp_allocator = NULL;
static PhysicsSystem *g_physics_system = NULL;
static BPLayerInterfaceImpl g_broad_phase_layer_interface;
static ObjectVsBroadPhaseLayerFilterImpl g_object_vs_broadphase_layer_filter;
static ObjectLayerPairFilterImpl g_object_vs_object_layer_filter;

typedef struct HpVoxelNeighbours
{
	uint32_t neighbours;
} HpVoxelNeighbours;

typedef struct PhysicsAs
{
	uint32_t corners_len;
	UVec4* corners;
	uint32_t neighbours_len;
	HpVoxelNeighbours* neighbours;
	Vec3 grid_size;
} PhysicsAs;

class VoxelShape final : public Shape
{
public:
    JPH_OVERRIDE_NEW_DELETE

    VoxelShape(const PhysicsMaterial *inMaterial, const PhysicsAs *physics_as) : Shape(EShapeType::User1, EShapeSubType::User1)
    {
        this->physics_as = physics_as;
        this->mMaterial = inMaterial;
    }

    // Register shape functions with the registry
    static void sRegister()
    {
        printf("Registering VoxelShape\n");

        CollisionDispatch::sRegisterCollideShape(EShapeSubType::User1, EShapeSubType::User1, collideSelf);
    }

    bool IsVoxelEmpty(Vec3 position) const
    {
        return GetVoxelNeighbours(position) == 0;
    }

    uint32_t GetVoxelIndex(Vec3 position) const
    {
        return (uint32_t)position.GetX() + (uint32_t)position.GetY() * physics_as->grid_size.GetX() + (uint32_t)position.GetZ() * physics_as->grid_size.GetX() * physics_as->grid_size.GetY();
    }

    uint32_t GetVoxelNeighbours(Vec3 position) const
    {
        if (position.GetX() < 0 || position.GetY() < 0 || position.GetZ() < 0)
            return true;

        if (position.GetX() >= physics_as->grid_size.GetX() || position.GetY() >= physics_as->grid_size.GetY() || position.GetZ() >= physics_as->grid_size.GetZ())
            return true;

        uint voxel_x = (uint)position.GetX();
        uint voxel_y = (uint)position.GetY();
        uint voxel_z = (uint)position.GetZ();

        uint voxel_index = voxel_x + voxel_y * physics_as->grid_size.GetX() + voxel_z * physics_as->grid_size.GetX() * physics_as->grid_size.GetY();
        auto voxel = physics_as->neighbours[voxel_index];
        return voxel.neighbours;
    }

    static void collideSelf(const Shape *inShape1, const Shape *inShape2, Vec3Arg inScale1, Vec3Arg inScale2, Mat44Arg inCenterOfMassTransform1, Mat44Arg inCenterOfMassTransform2, const SubShapeIDCreator &inSubShapeIDCreator1, const SubShapeIDCreator &inSubShapeIDCreator2, const CollideShapeSettings &inCollideShapeSettings, CollideShapeCollector &ioCollector, const ShapeFilter &inShapeFilter)
    {
        const VoxelShape *shape1 = static_cast<const VoxelShape *>(inShape1);
        const VoxelShape *shape2 = static_cast<const VoxelShape *>(inShape2);

        uint32_t max_voxels1 = shape1->physics_as->grid_size.GetX() + shape1->physics_as->grid_size.GetY() + shape1->physics_as->grid_size.GetZ();
        uint32_t max_voxels2 = shape2->physics_as->grid_size.GetX() + shape2->physics_as->grid_size.GetY() + shape2->physics_as->grid_size.GetZ();
    }

    virtual bool MustBeStatic() const override { return true; }
    // See Shape::GetLocalBounds
    virtual AABox GetLocalBounds() const override
    {
        return AABox(
            -this->physics_as->grid_size / 2,
            this->physics_as->grid_size / 2);
    }

    // See Shape::GetSubShapeIDBitsRecursive
    virtual uint GetSubShapeIDBitsRecursive() const override { return 0; }

    virtual float GetInnerRadius() const override { return 0.0f; }

    // See Shape::GetMassProperties
    virtual MassProperties GetMassProperties() const override
    {
        float mass = 1;
        // for (uint32_t i = 0; i < num_voxels; i++)
        // {
        //     HpVoxel voxel = voxels[i];
        //     if (voxel.ty == 0)
        //     {
        //         continue;
        //     }
        //     // density is just now 0.1;
        //     mass += 0.1;
        // }

        MassProperties prop;
        prop.ScaleToMass(mass);

        return prop;
    }

    // See Shape::GetMaterial
    virtual const PhysicsMaterial *GetMaterial(const SubShapeID &inSubShapeID) const override
    {
        JPH_ASSERT(inSubShapeID.IsEmpty(), "Invalid subshape ID");
        return mMaterial != nullptr ? mMaterial : PhysicsMaterial::sDefault;
    }

    // See Shape::GetSurfaceNormal
    virtual Vec3 GetSurfaceNormal(const SubShapeID &inSubShapeID, Vec3Arg inLocalSurfacePosition) const override
    {
        JPH_ASSERT(inSubShapeID.IsEmpty(), "Invalid subshape ID");
        return Vec3(0, 1, 0);
    }

    // See Shape::GetSupportingFace
    virtual void GetSupportingFace(const SubShapeID &inSubShapeID, Vec3Arg inDirection, Vec3Arg inScale, Mat44Arg inCenterOfMassTransform, SupportingFace &outVertices) const override
    {
    }
    // See Shape::CastRay
    virtual bool CastRay(const RayCast &inRay, const SubShapeIDCreator &inSubShapeIDCreator, RayCastResult &ioHit) const override
    {
        return false;
    }
    virtual void CastRay(const RayCast &inRay, const RayCastSettings &inRayCastSettings, const SubShapeIDCreator &inSubShapeIDCreator, CastRayCollector &ioCollector, const ShapeFilter &inShapeFilter = {}) const override
    {
    }

    // See: Shape::CollidePoint
    virtual void CollidePoint(Vec3Arg inPoint, const SubShapeIDCreator &inSubShapeIDCreator, CollidePointCollector &ioCollector, const ShapeFilter &inShapeFilter = {}) const override
    {
    }

    // See: Shape::CollideSoftBodyVertices
    virtual void CollideSoftBodyVertices(Mat44Arg inCenterOfMassTransform, Vec3Arg inScale, SoftBodyVertex *ioVertices, uint inNumVertices, float inDeltaTime, Vec3Arg inDisplacementDueToGravity, int inCollidingShapeIndex) const override
    {
    }

    // See Shape::GetTrianglesStart
    virtual void GetTrianglesStart(GetTrianglesContext &ioContext, const AABox &inBox, Vec3Arg inPositionCOM, QuatArg inRotation, Vec3Arg inScale) const override
    {
    }

    // See Shape::GetTrianglesNext
    virtual int GetTrianglesNext(GetTrianglesContext &ioContext, int inMaxTrianglesRequested, Float3 *outTriangleVertices, const PhysicsMaterial **outMaterials = nullptr) const override
    {
        return 0;
    }

    // See Shape::GetSubmergedVolume
    virtual void GetSubmergedVolume(Mat44Arg inCenterOfMassTransform, Vec3Arg inScale, const Plane &inSurface, float &outTotalVolume, float &outSubmergedVolume, Vec3 &outCenterOfBuoyancy JPH_IF_DEBUG_RENDERER(, RVec3Arg inBaseOffset)) const override { JPH_ASSERT(false, "Not supported"); }

    // See Shape
    // virtual void SaveBinaryState(StreamOut &inStream) const override;
    // virtual void SaveMaterialState(PhysicsMaterialList &outMaterials) const override;
    // virtual void RestoreMaterialState(const PhysicsMaterialRefC *inMaterials, uint inNumMaterials) override;

    // See Shape::GetStats
    virtual Stats GetStats() const override { return Stats(sizeof(*this), 0); }

    // See Shape::GetVolume
    virtual float GetVolume() const override { return 0; }

private:
    RefConst<PhysicsMaterial> mMaterial;
    const PhysicsAs *physics_as;
};

class VoxelShapeSettings final : public ShapeSettings
{
public:
    // JPH_DECLARE_SERIALIZABLE_VIRTUAL(JPH_EXPORT, VoxelShapeSettings)

    // /// Default constructor for deserialization
    // VoxelShapeSettings() = default;

    /// Create a plane shape.
    VoxelShapeSettings(const PhysicsMaterial *inMaterial, const PhysicsAs *inPhysics_as)
    {
        this->physics_as = inPhysics_as;
        this->mMaterial = inMaterial;
    }

    // See: ShapeSettings
    virtual ShapeResult Create() const override
    {
        if (mCachedResult.IsEmpty())
        {
            Ref<Shape> shape = new VoxelShape(this->mMaterial, this->physics_as);
            this->mCachedResult.Set(shape);
        }

        return mCachedResult;
    }

    const PhysicsMaterial *mMaterial = nullptr;
    const PhysicsAs *physics_as = nullptr;
};


void physics_init()
{
  // Register allocation hook. In this example we'll just let Jolt use malloc / free but you can override these if you want (see Memory.h).
  // This needs to be done before any other Jolt function is called.
  RegisterDefaultAllocator();

  // Install trace and assert callbacks
  Trace = TraceImpl;
  JPH_IF_ENABLE_ASSERTS(AssertFailed = AssertFailedImpl;)

  // Create a factory, this class is responsible for creating instances of classes based on their name or hash and is mainly used for deserialization of saved data.
  // It is not directly used in this example but still required.
  Factory::sInstance = new Factory();

  // Register all physics types with the factory and install their collision handlers with the CollisionDispatch class.
  // If you have your own custom shape types you probably need to register their handlers with the CollisionDispatch before calling this function.
  // If you implement your own default material (PhysicsMaterial::sDefault) make sure to initialize it before this function or else this function will create one for you.
  RegisterTypes();
  VoxelShape::sRegister();

  // We need a temp allocator for temporary allocations during the physics update. We're
  // pre-allocating 10 MB to avoid having to do allocations during the physics update.
  // B.t.w. 10 MB is way too much for this example but it is a typical value you can use.
  // If you don't want to pre-allocate you can also use TempAllocatorMalloc to fall back to
  // malloc / free.
  g_temp_allocator = new TempAllocatorImpl(10 * 1024 * 1024);

  // We need a job system that will execute physics jobs on multiple threads. Typically
  // you would implement the JobSystem interface yourself and let Jolt Physics run on top
  // of your own job scheduler. JobSystemThreadPool is an example implementation.
  g_job_system = new JobSystemThreadPool(cMaxPhysicsJobs, cMaxPhysicsBarriers, thread::hardware_concurrency() - 1);

  // This is the max amount of rigid bodies that you can add to the physics system. If you try to add more you'll get an error.
  // Note: This value is low because this is a simple test. For a real project use something in the order of 65536.
  const uint cMaxBodies = 1024;

  // This determines how many mutexes to allocate to protect rigid bodies from concurrent access. Set it to 0 for the default settings.
  const uint cNumBodyMutexes = 0;

  // This is the max amount of body pairs that can be queued at any time (the broad phase will detect overlapping
  // body pairs based on their bounding boxes and will insert them into a queue for the narrowphase). If you make this buffer
  // too small the queue will fill up and the broad phase jobs will start to do narrow phase work. This is slightly less efficient.
  // Note: This value is low because this is a simple test. For a real project use something in the order of 65536.
  const uint cMaxBodyPairs = 1024;

  // This is the maximum size of the contact constraint buffer. If more contacts (collisions between bodies) are detected than this
  // number then these contacts will be ignored and bodies will start interpenetrating / fall through the world.
  // Note: This value is low because this is a simple test. For a real project use something in the order of 10240.
  const uint cMaxContactConstraints = 1024;

  // Create mapping table from object layer to broadphase layer
  // Note: As this is an interface, PhysicsSystem will take a reference to this so this instance needs to stay alive!
  // BPLayerInterfaceImpl broad_phase_layer_interface;

  // // Create class that filters object vs broadphase layers
  // // Note: As this is an interface, PhysicsSystem will take a reference to this so this instance needs to stay alive!
  // ObjectVsBroadPhaseLayerFilterImpl object_vs_broadphase_layer_filter;

  // // Create class that filters object vs object layers
  // // Note: As this is an interface, PhysicsSystem will take a reference to this so this instance needs to stay alive!
  // ObjectLayerPairFilterImpl object_vs_object_layer_filter;

  // Now we can create the actual physics system.
  g_physics_system = new PhysicsSystem();
  g_physics_system->Init(cMaxBodies, cNumBodyMutexes, cMaxBodyPairs, cMaxContactConstraints, g_broad_phase_layer_interface, g_object_vs_broadphase_layer_filter, g_object_vs_object_layer_filter);

  // A body activation listener gets notified when bodies activate and go to sleep
  // Note that this is called from a job so whatever you do here needs to be thread safe.
  // Registering one is entirely optional.
  // MyBodyActivationListener body_activation_listener;
  // g_physics_system.SetBodyActivationListener(&body_activation_listener);

  // A contact listener gets notified when bodies (are about to) collide, and when they separate again.
  // Note that this is called from a job so whatever you do here needs to be thread safe.
  // Registering one is entirely optional.
  // MyContactListener contact_listener;
  // g_physics_system.SetContactListener(&contact_listener);

  // The main way to interact with the bodies in the physics system is through the body interface. There is a locking and a non-locking
  // variant of this. We're going to use the locking version (even though we're not planning to access bodies from multiple threads)
  // BodyInterface &body_interface = g_physics_system->GetBodyInterface();

  // // Next we can create a rigid body to serve as the floor, we make a large box
  // // Create the settings for the collision volume (the shape).
  // // Note that for simple shapes (like boxes) you can also directly construct a BoxShape.
  // BoxShapeSettings floor_shape_settings(Vec3(100.0f, 1.0f, 100.0f));
  // floor_shape_settings.SetEmbedded(); // A ref counted object on the stack (base class RefTarget) should be marked as such to prevent it from being freed when its reference count goes to 0.

  // // Create the shape
  // ShapeSettings::ShapeResult floor_shape_result = floor_shape_settings.Create();
  // ShapeRefC floor_shape = floor_shape_result.Get(); // We don't expect an error here, but you can check floor_shape_result for HasError() / GetError()

  // // Create the settings for the body itself. Note that here you can also set other properties like the restitution / friction.
  // BodyCreationSettings floor_settings(floor_shape, RVec3(0.0f, -1.0f, 0.0f), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING);

  // // Create the actual rigid body
  // Body *floor = body_interface.CreateBody(floor_settings); // Note that if we run out of bodies this can return nullptr

  // // Add it to the world
  // body_interface.AddBody(floor->GetID(), EActivation::DontActivate);

  // // Now create a dynamic body to bounce on the floor
  // // Note that this uses the shorthand version of creating and adding a body to the world
  // BodyCreationSettings sphere_settings(new SphereShape(0.5f), RVec3(0.0f, 2.0f, 0.0f), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING);
  // g_sphere_body_id = body_interface.CreateAndAddBody(sphere_settings, EActivation::Activate);

  // // Now you can interact with the dynamic body, in this case we're going to give it a velocity.
  // // (note that if we had used CreateBody then we could have set the velocity straight on the body before adding it to the physics system)
  // body_interface.SetLinearVelocity(g_sphere_body_id, Vec3(0.0f, -5.0f, 0.0f));

  // Optional step: Before starting the physics simulation you can optimize the broad phase. This improves collision detection performance (it's pointless here because we only have 2 bodies).
  // You should definitely not call this every frame or when e.g. streaming in a new level section as it is an expensive operation.
  // Instead insert all new objects in batches instead of 1 at a time to keep the broad phase efficient.
  // g_physics_system->OptimizeBroadPhase();
}

BodyID create_box(Vec3 size, RVec3 position, Quat rotation)
{
  Ref<BoxShapeSettings> box_shape_settings = new BoxShapeSettings(size);

  // Create the shape
  BodyInterface &body_interface = g_physics_system->GetBodyInterface();
  BodyCreationSettings body_settings(box_shape_settings, position, rotation, EMotionType::Dynamic, Layers::MOVING);
  BodyID body_id = body_interface.CreateAndAddBody(body_settings, EActivation::Activate);
  return body_id;
}

struct BodyAndModel
{
  BodyID body_id;
  RL::Model model;
  RL::Color color = generate_color();

  static RL::Color generate_color()
  {
    return {
      static_cast<unsigned char>(RL::GetRandomValue(0, 255)),
      static_cast<unsigned char>(RL::GetRandomValue(0, 255)),
      static_cast<unsigned char>(RL::GetRandomValue(0, 255)),
      255
    };
  }

  static BodyAndModel create_box(Vec3 size, RVec3 position, Quat rotation, bool is_static = false)
  {
    
    Ref<BoxShapeSettings> box_shape_settings = new BoxShapeSettings(size);

    // Create the shape
    BodyInterface &body_interface = g_physics_system->GetBodyInterface();
    BodyCreationSettings body_settings(box_shape_settings, position, rotation, is_static ? EMotionType::Static : EMotionType::Dynamic, Layers::MOVING);
    BodyID body_id = body_interface.CreateAndAddBody(body_settings, EActivation::Activate);
  
    RL::Model model = RL::LoadModelFromMesh(RL::GenMeshCube(size.GetX(), size.GetY(), size.GetZ()));
    return {body_id, model};
  }
};

int main(int argc, const char *argv[])
{
  physics_init();
  RL::SetRandomSeed(0);
  RL::InitWindow(800, 450, "Jolt Physics Library");

  RL::SetTargetFPS(60);

  RL::Camera3D camera = {0};
  camera.position = {10.0f, 10.0f, 10.0f};
  camera.target = {0.0f, 0.0f, 0.0f};
  camera.up = {0.0f, 1.0f, 0.0f};
  camera.fovy = 45.0f;
  camera.projection = RL::CAMERA_PERSPECTIVE;  

  Array<BodyAndModel> bodies;
  bodies.push_back(
    BodyAndModel::create_box(Vec3(1.0f, 1.0f, 1.0f), RVec3(0.0f, 5.0f, 0.0f), Quat::sIdentity())
  );
  bodies.push_back(
    BodyAndModel::create_box(Vec3(1.0f, 1.0f, 1.0f), RVec3(0.0f, 5.0f, 0.0f), Quat::sEulerAngles(Vec3(45.0f, 45.0f, 0.0f)))
  );
  bodies.push_back(
    BodyAndModel::create_box(Vec3(10.0f, 1.0f, 10.0f), RVec3(0.0f, 0.0f, 0.0f), Quat::sIdentity(), true)
  );

  while (!RL::WindowShouldClose())
  {
    const int cCollisionSteps = 1;
    g_physics_system->Update(1.0f / 60.0f, cCollisionSteps, g_temp_allocator, g_job_system);

    RL::UpdateCamera(&camera, RL::CAMERA_ORBITAL);    

    RL::BeginDrawing();
    RL::ClearBackground(RL::RAYWHITE);

    RL::BeginMode3D(camera);

    for (BodyAndModel body : bodies)
    {
      BodyInterface &body_interface = g_physics_system->GetBodyInterface();
      const Vec3 &position = body_interface.GetPosition(body.body_id);
      const Quat &rotation = body_interface.GetRotation(body.body_id);

      RL::DrawModelEx(body.model, {position.GetX(), position.GetY(), position.GetZ()}, {rotation.GetX(), rotation.GetY(), rotation.GetZ()}, rotation.GetW(), {1.0f, 1.0f, 1.0f}, body.color);
    }

    RL::EndMode3D();

    // RL::DrawText("Congrats! You created your first window!", 190, 200, 20, RL::LIGHTGRAY);
    RL::EndDrawing();
  }

  return 0;
}
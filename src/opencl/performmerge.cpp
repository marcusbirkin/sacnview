#include "performmerge.h"
#include "sacnlistener.h"

using namespace openCL_experimental;

clMergeWorker *clMerge::workerThread = Q_NULLPTR;

// TODO At somepoint, pick the best, not just the first...
static const unsigned int deviceNum = 0;
static const unsigned int platformNum = 0;

static const auto changedUniversesForcedInterval = std::chrono::milliseconds(E131_NETWORK_DATA_LOSS_TIMEOUT);

static const cl::STRING_CLASS source = {
    /* performMerge
     *
     * [Input/Output] uchar* mergedLevels
     *  - 2D Array of merged levels
     *  - Host passes previous levels for use with changedUniverses
     *  - Kernel returnes merged universe
     *  - uchar mergedLevels[universe][slot]
     *
     * [Output] ulong* mergedLevelsSourceCIDHi
     * - 3D Array of merged level source CID (Hi Word)
     * - ulong mergedLevelsSourceCIDHi[universe][slot];
     *
     * [Output] ulong* mergedLevelsSourceCIDLo
     * - 3D Array of merged level source CID (Lo Word)
     * - ulong mergedLevelsSourceCIDLo[universe][slot];
     *
     * [Input/Output] bool* changedUniverses
     * - Array of changed universes
     * - Cleared by host and set by kernel
     * - bool changedUniverses[universe]
     *
     * [Input] uchar* sourceCounts
     * - Array of sources per universe
     * - uchar sourceCounts[universe];
     *
     * [Input] uchar* sourceLevels
     *  - 3D Array of source levels
     *  - uchar sourceLevels[universe][source][slot];
     *
     * [Input] uchar* sourcePriorities
     * - 3D Array of source priorities
     * - uchar sourcePriorities[universe][source][slot];
     *
     * [Input] ulong* sourceCIDsHi
     * - 2D Array of source CIDs (Hi Word)
     * - ulong sourceCIDsHi[universe][source];
     *
     * [Input] ulong* sourceCIDsLo
     * - 2D Array of source CIDs (Lo Word)
     * - ulong sourceCIDsLo[universe][source];
     *
     * Work item dimension 0
     * - Universe
     *
     * Work item dimension 1
     * - Slot
     *
     */

    R"(__kernel void performMerge (
            __global uchar* mergedLevels,
            __global ulong* mergedLevelsSourceCIDHi,
            __global ulong* mergedLevelsSourceCIDLo,
            __global bool* changedUniverses,
            __global uchar* sourceCounts,
            __global uchar* sourceLevels,
            __global uchar* sourcePriorities,
            __global ulong* sourceCIDsHi,
            __global ulong* sourceCIDsLo
        ) {
            // This work item's Universe/Slot
            size_t universe = get_global_id(0);
            size_t slot = get_global_id(1);

            // Universe/Slot counts
            size_t universeCount = get_global_size(0);
            size_t slotCount = get_global_size(1);

            // Find correct indexes within flattened array
            uint mergedIdx = slot + (universe * slotCount);
            uint slotIdxStart = slot;
            uint sourceIdxStart = 0;
            for (uint n = 0; n < universe; ++n) {
                slotIdxStart += sourceCounts[n] * slotCount;
                sourceIdxStart += sourceCounts[n];
            }
            uint slotIdxEnd = slotIdxStart + (sourceCounts[universe] * slotCount);


            // Find winning source
            uint winningSlotIdx = slotIdxStart;
            uint winningSourceIdx = sourceIdxStart;
            uint sourceIdx = sourceIdxStart;
            for (uint slotIdx = slotIdxStart; slotIdx < slotIdxEnd; slotIdx += slotCount) {
                if (sourcePriorities[slotIdx] > sourcePriorities[winningSlotIdx]) {
                    // Higher priority
                    winningSlotIdx = slotIdx;
                    winningSourceIdx = sourceIdx;
                } else if (sourcePriorities[slotIdx] == sourcePriorities[winningSlotIdx]) {
                    // Same priority
                    if (sourceLevels[slotIdx] > sourceLevels[winningSlotIdx]) {
                        //...But higher level
                        winningSlotIdx = slotIdx;
                        winningSourceIdx = sourceIdx;
                    }
                }
                ++sourceIdx;
            }

            // Flag if universe has changed
            // Don't be tempted to use |=, remember multiple threads accessing
            if (mergedLevels[mergedIdx] != sourceLevels[winningSlotIdx])
                changedUniverses[universe] = true;

            // Finally return winning level and it's source
            mergedLevels[mergedIdx] = sourceLevels[winningSlotIdx];
            mergedLevelsSourceCIDHi[mergedIdx] = sourceCIDsHi[winningSourceIdx];
            mergedLevelsSourceCIDLo[mergedIdx] = sourceCIDsLo[winningSourceIdx];
        }
    )"
};

clMerge::clMerge(QObject *parent) : QObject(parent)
{
    if (!workerThread) {
        workerThread = new clMergeWorker(this);
        workerThread->setObjectName("OpenCL Merge");
        connect(workerThread, &clMergeWorker::threadAborted, workerThread, &QObject::deleteLater);
        connect(workerThread, &clMergeWorker::finished, workerThread, &QObject::deleteLater);
        connect(workerThread, &clMergeWorker::destroyed, this, [=]() { workerThread = Q_NULLPTR; });
        workerThread->start();
    }
}

clMerge::~clMerge() {
    workerThread->quit();
    workerThread->wait();
}

bool clMerge::isRunning() {
    if (!workerThread) return false;
    return workerThread->isRunning();
}

unsigned int clMerge::getMergesPerSec() {
    if (!workerThread) return 0;
    return workerThread->mergesPerSec;
}

void clMerge::setSourceLevels(
        uint16_t universe, CID cid,
        slotData_t::const_iterator sourceBegin,
        slotData_t::const_iterator sourceEnd)
{
    if (!workerThread)
        return;

    auto universeIdx = clMergeWorker::universe2Idx(universe);
    if (universeIdx == -1)
        return;

    auto sourceIdx = clMergeWorker::source2Idx(universe, cid);
    if (sourceIdx == -1)
        return;

    if (!workerThread->sourceLevels)
        return;

    auto dest = workerThread->sourceLevels->universePtr(universeIdx, sourceIdx);
    if (dest)
        std::copy(sourceBegin, sourceEnd, dest->begin());

    auto cidPair = clMergeWorker::CID2Ulong(cid);
    workerThread->sourceCIDsHi->universe(universeIdx, sourceIdx) = cidPair.first;
    workerThread->sourceCIDsLo->universe(universeIdx, sourceIdx) = cidPair.second;
}

void clMerge::setSourcePriorities(
        uint16_t universe, CID cid,
        slotData_t::const_iterator sourceBegin,
        slotData_t::const_iterator sourceEnd)
{
    if (!workerThread)
        return;

    auto universeIdx = clMergeWorker::universe2Idx(universe);
    if (universeIdx == -1)
        return;

    auto sourceIdx = clMergeWorker::source2Idx(universe, cid);
    if (sourceIdx == -1)
        return;

    if (!workerThread->sourcePriorities)
        return;

    auto dest = workerThread->sourcePriorities->universePtr(universeIdx, sourceIdx);
    if (dest)
        std::copy(sourceBegin, sourceEnd, dest->begin());
}

CID clMerge::winningSource(uint16_t universe, uint16_t slot) {
    if (!workerThread)
        return CID();
    if (slot >= DMX_SLOT_MAX)
        return CID();

    auto universeIdx = clMergeWorker::universe2Idx(universe);
    if (universeIdx == -1)
        return CID();
    if (universeIdx >= workerThread->mergedLevelsSourceCIDHi->getUniverseCount() ||
            universeIdx >= workerThread->mergedLevelsSourceCIDLo->getUniverseCount())
        return CID();

    return clMergeWorker::ulong2CID(
                workerThread->mergedLevelsSourceCIDHi->universe(universeIdx).at(slot),
                workerThread->mergedLevelsSourceCIDLo->universe(universeIdx).at(slot));
}

cl_int clMergeWorker::init() {
    cl_int status = CL_SUCCESS;

    // Get platform
    std::vector<cl::Platform> platforms;
    status = cl::Platform::get(&platforms);
    if (status != CL_SUCCESS)
        msgErr("cl::Platform::get", status);
    if (platforms.size() == 0) {
        qDebug() << "CL status: No platforms";
        return CL_DEVICE_NOT_AVAILABLE;
    }
    for (auto platform : platforms)
        qDebug() << "CL: Found Platform:" << platform.getInfo<CL_PLATFORM_NAME>().c_str()
                 << "Version:" << platform.getInfo<CL_PLATFORM_VERSION>().c_str();

    // Get device and create context
    std::vector<cl::Device> devices;
    status = platforms[platformNum].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (status != CL_SUCCESS)
        msgErr("cl::Platform::getDevices", status);
    if (devices.size() == 0) {
        qDebug() << "CL status: No devices";
        return CL_DEVICE_NOT_AVAILABLE;
    }
    context = cl::Context(devices, NULL, NULL, NULL, &status);
    if (status != CL_SUCCESS)
        msgErr("cl::Context", status);

    for (auto device : devices)
        qDebug() << "CL: Found GPU Device:" << device.getInfo<CL_DEVICE_NAME>().c_str()
                 << "Version:" << device.getInfo<CL_DEVICE_VERSION>().c_str();

    // Load and build program
    cl::Program program;
    program = cl::Program(context, source, CL_TRUE, &status);
    if (status != CL_SUCCESS)
        msgErr("cl::Program", status,
               QString("CL_BUILD_PROGRAM_FAILURE\n\n%1")
                              .arg(program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[deviceNum]).c_str()));


    // Create kernel
    kernel = cl::Kernel(program, "performMerge", &status);
    if (status != CL_SUCCESS)
        msgErr("cl::Kernel", status);

    // Setup Buffers
    status = setupBuffers();
    if (status != CL_SUCCESS)
        msgErr("setupBuffers()", status);

    return status;
};

CID clMergeWorker::ulong2CID(ulong hi, ulong lo) {
    CID cid;
    std::array<uint8_t, CID::CIDBYTES> cidArr;
    memcpy(cidArr.data(), &hi, cidArr.size() / 2);
    memcpy(cidArr.data() + (cidArr.size() / 2), &lo, cidArr.size() / 2);
    cid.Unpack(cidArr.data());

    return cid;
}

std::pair<ulong /*hi*/, ulong /*lo*/> clMergeWorker::CID2Ulong(CID &cid) {
    std::array<uint8_t, CID::CIDBYTES> cidArr;
    std::pair<ulong, ulong> ret;
    cid.Pack(cidArr.data());
    memcpy(&ret.first, cidArr.data(), cidArr.size() / 2);
    memcpy(&ret.second, cidArr.data() + (cidArr.size() / 2), cidArr.size() / 2);

    return ret;
}

int clMergeWorker::universe2Idx(uint16_t universe) {
    return sACNManager::getInstance()->getListenerList().keys().indexOf(universe);
}

int clMergeWorker::source2Idx(uint16_t universe, CID cid) {
    auto sourceIdx = 0;
    for (auto source : sACNManager::getInstance()->getListener(universe)->getSourceList()) {
        if (source->src_cid == cid)
            return sourceIdx;

        ++sourceIdx;
    }

    return -1;
}

uint16_t clMergeWorker::idx2Universe(size_t idx) {
    auto list = sACNManager::getInstance()->getListenerList().values();
    if (static_cast<size_t>(list.size()) > idx) {
        auto listener = list.at(idx).toStrongRef();
        if (listener)
            return listener->universe();
    }
    return 0;
}

const sACNSource *clMergeWorker::idx2Source(uint16_t universe, size_t idx) {
    if (idx >= sACNManager::getInstance()->getListener(universe)->getSourceList().size())
        return Q_NULLPTR;
    return sACNManager::getInstance()->getListener(universe)->getSourceList().at(idx);
}

void clMergeWorker::msgErr(QString caller, cl_int err, QString details) {
    QString errStr;
    switch (err) {
        case CL_BUILD_PROGRAM_FAILURE:
            errStr = QString("CL_BUILD_PROGRAM_FAILURE");
            break;

        case CL_INVALID_COMMAND_QUEUE:
            errStr = QString("CL_INVALID_COMMAND_QUEUE");
            break;

        case CL_INVALID_MEM_OBJECT:
            errStr = QString("CL_INVALID_MEM_OBJECT");
            break;

        case CL_INVALID_PROGRAM_EXECUTABLE:
            errStr = QString("CL_INVALID_PROGRAM_EXECUTABLE");
            break;

        case CL_INVALID_KERNEL_NAME:
            errStr = QString("CL_INVALID_KERNEL_NAME");
        break;

        case CL_INVALID_ARG_INDEX:
            errStr = QString("CL_INVALID_ARG_INDEX");
        break;

        case CL_INVALID_ARG_SIZE:
            errStr = QString("CL_INVALID_ARG_SIZE");
        break;

        case CL_INVALID_KERNEL_ARGS:
            errStr = QString("CL_INVALID_KERNEL_ARGS");
        break;

        case CL_INVALID_WORK_GROUP_SIZE:
            errStr = QString("CL_INVALID_WORK_GROUP_SIZE");
        break;

        case CL_INVALID_EVENT:
            errStr = QString("CL_INVALID_EVENT");
        break;

        case CL_INVALID_BUFFER_SIZE:
            errStr = QString("CL_INVALID_BUFFER_SIZE");
        break;

        case CL_INVALID_GLOBAL_WORK_SIZE:
            errStr = QString("CL_INVALID_GLOBAL_WORK_SIZE");
        break;

        default:
            errStr = QString::number(err);
            break;
    }

    QString detailsLine;
    QTextStream stream(&details);

    qDebug().noquote() << "CL status" << caller << errStr;
    while (stream.readLineInto(&detailsLine))
        qDebug().noquote() << "CL status" << detailsLine;
}

cl_int clMergeWorker::setupBuffers() {
    cl_int status = CL_SUCCESS;

    auto universeCount = sACNManager::getInstance()->getListenerList().count();

    // [Input/Output] uchar* mergedLevels
    if (!mergedLevels)
        mergedLevels = new bufferUniverseData<clMerge::slotData_t>(
                    context,
                    CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                    this);
    status = mergedLevels->setUniverseCount(universeCount);
    if (status != CL_SUCCESS) {
        msgErr("setUniverseCount()", status, QString("mergedLevels"));
        return status;
    }

    // [Output] ulong* mergedLevelsSourceCIDHi
    if (!mergedLevelsSourceCIDHi)
        mergedLevelsSourceCIDHi = new bufferUniverseData<std::array<uint64_t, DMX_SLOT_MAX>>(
                    context,
                    CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_USE_HOST_PTR,
                    this);
    status = mergedLevelsSourceCIDHi->setUniverseCount(universeCount);
    if (status != CL_SUCCESS) {
        msgErr("setUniverseCount()", status, QString("mergedLevelsSourceCIDHi"));
        return status;
    }

    // [Output] ulong* mergedLevelsSourceCIDLo
    if (!mergedLevelsSourceCIDLo)
        mergedLevelsSourceCIDLo = new bufferUniverseData<std::array<uint64_t, DMX_SLOT_MAX>>(
                    context,
                    CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_USE_HOST_PTR,
                    this);
    status = mergedLevelsSourceCIDLo->setUniverseCount(universeCount);
    if (status != CL_SUCCESS) {
        msgErr("setUniverseCount()", status, QString("mergedLevelsSourceCIDLo"));
        return status;
    }

    // [Input/Output] bool* changedUniverses
    if (!changedUniverses)
        changedUniverses = new bufferUniverseData<uint8_t>(
                    context,
                    CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                    this);
    status = changedUniverses->setUniverseCount(universeCount);
    if (status != CL_SUCCESS) {
        msgErr("setUniverseCount()", status, QString("changedUniverses"));
        return status;
    }

    // [Input] uchar* sourceCounts
    if (!sourceCounts)
        sourceCounts = new bufferUniverseData<uint8_t>(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                    this);
    status = sourceCounts->setUniverseCount(universeCount);
    if (status != CL_SUCCESS) {
        msgErr("setUniverseCount()", status, QString("sourceCounts"));
        return status;
    }

    // [Input] uchar* sourceLevels
    if (!sourceLevels)
        sourceLevels = new bufferSourceData<clMerge::slotData_t>(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                    this);
    status = sourceLevels->setUniverseCount(universeCount);
    status |= sourceLevels->setSourceCount(maxSources);
    if (status != CL_SUCCESS) {
        msgErr("setUniverseCount()/setSourceCount()", status, QString("sourceLevels"));
        return status;
    }

    // [Input] uchar* sourcePriorities
    if (!sourcePriorities)
        sourcePriorities = new bufferSourceData<clMerge::slotData_t>(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                    this);
    status = sourcePriorities->setUniverseCount(universeCount);
    status |= sourcePriorities->setSourceCount(maxSources);
    if (status != CL_SUCCESS) {
        msgErr("setUniverseCount()/setSourceCount()", status, QString("sourcePriorities"));
        return status;
    }

    // [Input] ulong* sourceCIDsHi
    if (!sourceCIDsHi)
        sourceCIDsHi = new bufferSourceData<uint64_t>(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                    this);
    status = sourceCIDsHi->setUniverseCount(universeCount);
    status |= sourceCIDsHi->setSourceCount(maxSources);
    if (status != CL_SUCCESS) {
        msgErr("setUniverseCount()/setSourceCount()", status, QString("sourceCIDsHi"));
        return status;
    }

    // [Input] ulong* sourceCIDsLo
    if (!sourceCIDsLo)
        sourceCIDsLo = new bufferSourceData<uint64_t>(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                    this);
    status = sourceCIDsLo->setUniverseCount(universeCount);
    status |= sourceCIDsLo->setSourceCount(maxSources);
    if (status != CL_SUCCESS) {
        msgErr("setUniverseCount()/setSourceCount()", status, QString("sourceCIDsLo"));
        return status;
    }

    return status;
}

void clMergeWorker::run() {
    cl_int status = init();

    while (status == CL_SUCCESS) {
        if (exitThread) return;

        // Resize, if needed, buffers
        status = setupBuffers();
        if (status != CL_SUCCESS) {
            msgErr("setupBuffers()", status);
            break;
        }

        // If there are no universes, don't bother
        if (mergedLevels->size() == 0) {
            QThread::sleep(1);
            continue;
        }

        // TODO Dynamic sizing based on actual source count
        sourceCounts->fill(maxSources);

        // Reset universe changed flag
        changedUniverses->fill(false);

        // Queue write buffers
        std::vector<cl::Event> writeBufferEvents;
        cl::CommandQueue queue = cl::CommandQueue(context, 0, &status);
        if (status != CL_SUCCESS) {
            msgErr("cl::CommandQueue", status);
            break;
        }

        writeBufferEvents.emplace_back();
        status = queue.enqueueWriteBuffer(
                    mergedLevels->buffer(),
                    CL_FALSE, 0,
                    mergedLevels->size(), mergedLevels->data(),
                    NULL, &writeBufferEvents.back());
        if (status != CL_SUCCESS) {
            msgErr("cl::Queue:enqueueWriteBuffer", status, "mergedLevels");
            break;
        }

        writeBufferEvents.emplace_back();
        status = queue.enqueueWriteBuffer(
                    changedUniverses->buffer(),
                    CL_FALSE, 0,
                    changedUniverses->size(), changedUniverses->data(),
                    NULL, &writeBufferEvents.back());
        if (status != CL_SUCCESS) {
            msgErr("cl::Queue:enqueueWriteBuffer", status, "changedUniverses");
            break;
        }

        writeBufferEvents.emplace_back();
        status = queue.enqueueWriteBuffer(
                    sourceLevels->buffer(),
                    CL_FALSE, 0,
                    sourceLevels->size(), sourceLevels->data(),
                    NULL, &writeBufferEvents.back());
        if (status != CL_SUCCESS) {
            msgErr("cl::Queue:enqueueWriteBuffer", status, "sourceLevels");
            break;
        }

        writeBufferEvents.emplace_back();
        status = queue.enqueueWriteBuffer(
                    sourcePriorities->buffer(),
                    CL_FALSE, 0,
                    sourcePriorities->size(), sourcePriorities->data(),
                    NULL, &writeBufferEvents.back());
        if (status != CL_SUCCESS) {
            msgErr("cl::Queue:enqueueWriteBuffer", status, "sourcePriorities");
            break;
        }

        writeBufferEvents.emplace_back();
        status = queue.enqueueWriteBuffer(
                    sourceCIDsHi->buffer(),
                    CL_FALSE, 0,
                    sourceCIDsHi->size(), sourceCIDsHi->data(),
                    NULL, &writeBufferEvents.back());
        if (status != CL_SUCCESS) {
            msgErr("cl::Queue:enqueueWriteBuffer", status, "sourceCIDsHi");
            break;
        }

        writeBufferEvents.emplace_back();
        status = queue.enqueueWriteBuffer(
                    sourceCIDsLo->buffer(),
                    CL_FALSE, 0,
                    sourceCIDsLo->size(), sourceCIDsLo->data(),
                    NULL, &writeBufferEvents.back());
        if (status != CL_SUCCESS) {
            msgErr("cl::Queue:enqueueWriteBuffer", status, "sourceCIDsLo");
            break;
        }

        // Set kernel arguments and queue
        unsigned int arg = 0;
        kernel.setArg(arg++, mergedLevels->buffer());
        kernel.setArg(arg++, mergedLevelsSourceCIDHi->buffer());
        kernel.setArg(arg++, mergedLevelsSourceCIDLo->buffer());
        kernel.setArg(arg++, changedUniverses->buffer());
        kernel.setArg(arg++, sourceCounts->buffer());
        kernel.setArg(arg++, sourceLevels->buffer());
        kernel.setArg(arg++, sourcePriorities->buffer());
        kernel.setArg(arg++, sourceCIDsHi->buffer());
        kernel.setArg(arg++, sourceCIDsLo->buffer());
        std::vector<cl::Event> kernelEvent(1);
        status = queue.enqueueNDRangeKernel(
                    kernel,
                    cl::NullRange,
                    cl::NDRange(mergedLevels->getUniverseCount(), DMX_SLOT_MAX),
                    cl::NullRange,
                    &writeBufferEvents, &kernelEvent.front());
        if (status != CL_SUCCESS) {
            msgErr("cl::Queue:enqueueNDRangeKernel", status);
            break;
        }

        // Queue read buffers
        std::vector<cl::Event> dataReadyEvents;
        dataReadyEvents.emplace_back();
        status = queue.enqueueReadBuffer(
                    mergedLevels->buffer(),
                    CL_FALSE, 0,
                    mergedLevels->size(), mergedLevels->data(),
                    &kernelEvent, &dataReadyEvents.back());
        if (status != CL_SUCCESS) {
            msgErr("cl::Queue:enqueueReadBuffer", status, "mergedLevels");
            break;
        }

        dataReadyEvents.emplace_back();
        status = queue.enqueueReadBuffer(
                    mergedLevelsSourceCIDHi->buffer(),
                    CL_FALSE, 0,
                    mergedLevelsSourceCIDHi->size(), mergedLevelsSourceCIDHi->data(),
                    &kernelEvent, &dataReadyEvents.back());
        if (status != CL_SUCCESS) {
            msgErr("cl::Queue:enqueueReadBuffer", status, "mergedLevelsSourceCIDHi");
            break;
        }

        dataReadyEvents.emplace_back();
        status = queue.enqueueReadBuffer(
                    mergedLevelsSourceCIDLo->buffer(),
                    CL_FALSE, 0,
                    mergedLevelsSourceCIDLo->size(), mergedLevelsSourceCIDLo->data(),
                    &kernelEvent, &dataReadyEvents.back());
        if (status != CL_SUCCESS) {
            msgErr("cl::Queue:enqueueReadBuffer", status, "mergedLevelsSourceCIDLo");
            break;
        }

        dataReadyEvents.emplace_back();
        queue.enqueueReadBuffer(
                        changedUniverses->buffer(),
                        CL_FALSE, 0,
                        changedUniverses->size(), changedUniverses->data(),
                        &kernelEvent, &dataReadyEvents.back());
        if (status != CL_SUCCESS) {
            msgErr("cl::Queue:enqueueReadBuffer", status, "changedUniverses");
            break;
        }

        // Flush queue
        status = queue.flush();
        if (status != CL_SUCCESS) {
            msgErr("cl::Queue:flush", status);
            break;
        }

        /* Wait until read buffers complete
         *
         * n.b. This is in place of cl:Finish(),
         * as that is a blocking method that consumes 100% CPU
         */
        cl_int statusReadBuffer;
        do {
            statusReadBuffer = 0;
            for (auto &event : dataReadyEvents)
                statusReadBuffer |= event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
            QThread::msleep(1);
        } while (statusReadBuffer != CL_COMPLETE);

        // Any data reported changed?
        if (changedUniversesForcedTimer.hasExpired(std::chrono::milliseconds(changedUniversesForcedInterval).count())) {
            // Every now and then, report a full update
            changedUniverses->fill(true);
            changedUniversesForcedTimer.restart();
        }
        for (unsigned int universeIdx = 0; universeIdx < mergedLevels->getUniverseCount(); ++universeIdx)
        {
            if (changedUniverses->universe(universeIdx)) {
                auto universe = idx2Universe(universeIdx);
                auto levels = mergedLevels->universePtr(universeIdx);
                if (universe && levels)
                    emit dataChanged(universe, levels);
            }
        }

        mergesPerSec.countMerge();

        // A modest pause, before we loop again
        QThread::msleep(10);
    }

    emit threadAborted();
    qDebug() << "CL Merger has aborted....";
}

void clMergeWorker::exit(int) {
    exitThread = true;
}

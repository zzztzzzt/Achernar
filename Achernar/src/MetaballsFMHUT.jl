module MetaballsFMHUT

import Fomalhaut as FMHUT

using ..MetaballsAX

export start_server

const _WS_PATH = "/metaballs"

function start_server()
    MetaballsAX.init!()
    app = FMHUT.App()
    cb_ptr, ctx_ptr = MetaballsAX.get_native_generator()
    @FMHUT.axis_websocket app _WS_PATH 120.0 cb_ptr ctx_ptr
    FMHUT.serve(app; fps=120)
end

end # module MetaballsFMHUT

if abspath(PROGRAM_FILE) == @__FILE__
    MetaballsFMHUT.start_server()
end
